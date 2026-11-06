# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh

def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1
    return 0.5 * (np.sum(adj) - s @ adj @ s)

def _spectral_init(adj, n):
    """Generate initial labels using spectral method (Fiedler vector)."""
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    try:
        evals, evecs = eigh(L, subset_by_index=[1, 1])
        fiedler = evecs[:, 0]
    except:
        return np.random.randint(0, 2, n).astype(np.float64)
    
    labels = (fiedler > 0).astype(np.float64)
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n).astype(np.float64)
    return labels

def _sdp_bcd_init(adj, n, deadline, max_iter=500):
    """
    Solve the Max-Cut SDP relaxation using Block Coordinate Descent (BCD).
    Maximize tr(A*W) s.t. W_ii = 1, W >= 0.
    
    Update rule for W_ij (i != j):
    We update the 2x2 submatrix W[[i,j],[i,j]] while keeping the rest fixed.
    However, a simpler and robust approach for BCD on PSD matrices with diag=1
    is to update one row i at a time.
    
    Let W = [w_ii  w_i^T; w_i  W_ii].
    We want to update w_i (vector of length n-1) to maximize the objective,
    subject to the 2x2 block [1, w_i^T; w_i, W_rest] being PSD? 
    Actually, standard BCD for SDP (Chen & Deng) updates one variable at a time 
    or a block. For Max-Cut, updating the entire row i is efficient.
    
    Let's use the update:
    w_i_new = argmax_{w_i} tr(A * W_new) subject to W_new >= 0 and (W_new)_ii = 1.
    
    The condition W_new >= 0 is equivalent to:
    The Schur complement of the (i,i) entry (which is 1) must be PSD.
    W_new = [[1, w_i^T], [w_i, M]] where M is the rest of W.
    Schur complement: M - w_i w_i^T >= 0.
    This implies w_i w_i^T <= M.
    
    The objective term involving w_i is:
    tr(A * W_new) = ... + 2 * sum_{j!=i} A_ij w_ij + A_ii (which is 0 usually, or fixed)
    Since A is symmetric and A_ii=0 for simple graphs, the part depending on w_i is:
    2 * (A_i_rest) @ w_i.
    
    So we want to maximize 2 * A_i_rest @ w_i subject to w_i w_i^T <= M.
    This is equivalent to maximizing c^T w subject to w^T w <= 1 and w in the ellipsoid defined by M?
    No, the constraint is w w^T <= M.
    
    Actually, a simpler and very effective heuristic for Max-Cut SDP BCD is:
    1. Initialize W = I.
    2. Iterate:
       For each i:
          Let u = A[:, i] (column i of A).
          Let M = W with row i and col i removed.
          We want to update w_i (the vector W[i, :]) to maximize u^T w_i subject to W_new >= 0.
          
          The constraint W_new >= 0 with W_ii=1 is equivalent to:
          w_i^T M^{-1} w_i <= 1 (if M is PD).
          If M is not PD, we can regularize or use eigenvalues.
          
          The solution to max u^T w s.t. w^T M^{-1} w <= 1 is:
          w* = M u / (u^T M u)^{1/2}  (scaled to satisfy the boundary).
          
          Let's compute:
          M is (n-1)x(n-1).
          We need M to be invertible/PSD.
          
          Algorithm:
          for i in range(n):
             idx = [j for j in range(n) if j != i]
             M = W[np.ix_(idx, idx)]
             u = A[np.ix_(idx, [i])] # vector of length n-1
             
             # Check if M is PD
             try:
                 # Solve M x = u
                 # If M is singular, add small regularization
                 x = np.linalg.solve(M + 1e-9*np.eye(len(idx)), u)
                 denom = np.sqrt(np.dot(u, x))
                 if denom < 1e-6:
                     continue
                 w_new = x / denom
             except:
                 continue
             
             # Update W
             W[i, idx] = w_new
             W[idx, i] = w_new
             W[i, i] = 1.0
    """
    W = np.eye(n, dtype=np.float64)
    A = adj.astype(np.float64)
    
    # Precompute indices for efficiency
    indices_list = [list(range(n)) for _ in range(n)]
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        for i in range(n):
            if time.time() > deadline - 10:
                break
            idx = [j for j in range(n) if j != i]
            if not idx:
                continue
            M = W[np.ix_(idx, idx)]
            u = A[np.ix_(idx, [i])].flatten()
            
            # Regularize M to ensure invertibility
            M_reg = M + 1e-6 * np.eye(len(idx))
            
            try:
                # We want to maximize u^T w subject to w^T M^{-1} w <= 1
                # Solution: w = M u / sqrt(u^T M u)
                # Note: M might not be exactly PD, but close.
                # We use solve to compute M^{-1} u
                x = np.linalg.solve(M_reg, u)
                denom = np.sqrt(np.dot(u, x))
                if denom < 1e-6:
                    continue
                w_new = x / denom
            except:
                continue
            
            W[i, idx] = w_new
            W[idx, i] = w_new
            W[i, i] = 1.0
    
    # Ensure W is PSD by clipping eigenvalues
    try:
        evals, evecs = eigh(W)
        evals = np.maximum(evals, 0)
        W = evecs @ np.diag(evals) @ evecs.T
        # Re-project diagonal to 1
        np.fill_diagonal(W, 1.0)
    except:
        pass
    
    # Goemans-Williamson Rounding
    # W = V V^T
    try:
        evals, evecs = eigh(W)
        evals = np.maximum(evals, 0)
        V = evecs * np.sqrt(evals)[None, :]
    except:
        return _spectral_init(adj, n)
    
    rng = np.random.RandomState(42)
    best_labels = None
    best_cut = -1
    
    # Try multiple random hyperplanes
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        u = rng.randn(n)
        u /= np.linalg.norm(u) + 1e-12
        labels = (V @ u > 0).astype(np.float64)
        
        if np.all(labels == 0) or np.all(labels == 1):
            continue
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = _spectral_init(adj, n)
    
    return best_labels

def _greedy_refine(adj, labels, deadline):
    """
    Vectorized best-improvement local search with incremental updates.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    As = adj @ s
    gains = s * As
    
    while True:
        if time.time() > deadline:
            break
        
        max_gain = np.max(gains)
        
        if max_gain <= 0:
            break
        
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        s[v] = -s[v]
        
        As += 2 * adj[:, v] * s[v]
        gains = s * As
    
    return labels, 0.5 * (np.sum(adj) - s @ adj @ s)

def _pair_flip_refine(adj, labels, deadline):
    """
    Local search allowing 2-flip moves.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    s = 2 * labels - 1
    current_cut = 0.5 * (np.sum(adj) - s @ adj @ s)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute degrees
    deg = np.sum(adj, axis=1)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all pairs? O(n^2) per pass. n=100 -> 10k pairs. 
        # We can vectorize the gain for flipping pair (i, j).
        # Gain of flipping i and j:
        # Delta = 2 * (s_i * A_ij + s_j * A_ij) - 2 * (s_i * (sum_{k!=j} A_ik s_k) + s_j * (sum_{k!=i} A_jk s_k)) ?
        # Let's derive:
        # Cut = 0.5 * (Sum(A) - s^T A s)
        # Flipping i, j: s_i' = -s_i, s_j' = -s_j.
        # Change in s^T A s:
        # (s^T A s)_new - (s^T A s)_old = 
        #   2 * s_i' (A s)_i + 2 * s_j' (A s)_j - 2 * s_i' s_j' A_ij - 2 * (s^T A s)_old? No.
        
        # Easier: 
        # Gain = 2 * (s_i * A_ij + s_j * A_ij) + 2 * (s_i * (As_i - A_ij s_j) + s_j * (As_j - A_ij s_i)) ?
        
        # Let's just compute gains for single flips first, if stuck, try pairs.
        # For simplicity and robustness within time limit, we'll do a limited pair search
        # on the vertices with highest single-flip potential.
        
        As = adj @ s
        gains_single = s * As
        
        # Find top k candidates for flipping
        k = min(20, n)
        top_idx = np.argsort(-gains_single)[:k]
        
        for i_idx in range(len(top_idx)):
            for j_idx in range(i_idx + 1, len(top_idx)):
                if time.time() > deadline - 5:
                    break
                i = top_idx[i_idx]
                j = top_idx[j_idx]
                
                # Calculate gain of flipping i and j
                # s_i, s_j flip
                # Delta(s^T A s) = 2 * s_i * (A s)_i + 2 * s_j * (A s)_j - 2 * s_i * s_j * A_ij * 2?
                
                # Let's use the formula:
                # New Cut = Old Cut + 2 * (s_i * (As_i - A_ij s_j) + s_j * (As_j - A_ij s_i) + A_ij * s_i * s_j * 2) ?
                
                # Correct formula for flipping set S:
                # Gain = 2 * sum_{v in S} s_v * (A s)_v - 2 * sum_{u,v in S, u!=v} A_uv s_u s_v
                # For S={i,j}:
                # Gain = 2 * s_i * (As_i) + 2 * s_j * (As_j) - 2 * A_ij * s_i * s_j * 2
                # Wait, (A s)_i includes A_ij s_j.
                # Let's expand:
                # s^T A s = sum_u sum_v A_uv s_u s_v
                # Flip i, j.
                # Terms involving i or j change.
                # Term i,i: A_ii s_i^2 (unchanged)
                # Term j,j: A_jj s_j^2 (unchanged)
                # Term i,j: A_ij s_i s_j -> A_ij (-s_i)(-s_j) = A_ij s_i s_j (unchanged)
                # Term i,k (k!=i,j): A_ik s_i s_k -> A_ik (-s_i) s_k = - A_ik s_i s_k
                # Term j,k (k!=i,j): A_jk s_j s_k -> A_jk (-s_j) s_k = - A_jk s_j s_k
                
                # So change is:
                # -2 * sum_{k!=i,j} A_ik s_i s_k - 2 * sum_{k!=i,j} A_jk s_j s_k
                # = -2 * s_i * (sum_{k!=i,j} A_ik s_k) - 2 * s_j * (sum_{k!=i,j} A_jk s_k)
                
                # We have As_i = sum_{k} A_ik s_k = A_ii s_i + A_ij s_j + sum_{k!=i,j} A_ik s_k
                # So sum_{k!=i,j} A_ik s_k = As_i - A_ii s_i - A_ij s_j
                # Assuming A_ii = 0:
                # sum_{k!=i,j} A_ik s_k = As_i - A_ij s_j
                
                # Gain = -2 * s_i * (As_i - A_ij s_j) - 2 * s_j * (As_j - A_ij s_i)
                # Gain = -2 * s_i * As_i + 2 * s_i * A_ij * s_j - 2 * s_j * As_j + 2 * s_j * A_ij * s_i
                # Gain = -2 * (s_i * As_i + s_j * As_j) + 4 * A_ij * s_i * s_j
                
                gain_i = -2 * s[i] * As[i]
                gain_j = -2 * s[j] * As[j]
                cross = 4 * adj[i, j] * s[i] * s[j]
                
                total_gain = gain_i + gain_j + cross
                
                if total_gain > 0:
                    # Perform flip
                    s[i] = -s[i]
                    s[j] = -s[j]
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    
                    # Update As
                    # As_k changes by -2 * A_ki * s_i_old - 2 * A_kj * s_j_old?
                    # As_new = As_old - 2 * A[:, i] * s_i_old - 2 * A[:, j] * s_j_old
                    # Note: s_i_old = -s_i_new
                    As -= 2 * adj[:, i] * (-s[i]) + 2 * adj[:, j] * (-s[j])
                    # Wait, As = A @ s.
                    # s changed at i, j.
                    # As_new = A @ s_new = A @ (s_old - 2*s_i_old*e_i - 2*s_j_old*e_j)
                    # As_new = As_old - 2*s_i_old*A[:,i] - 2*s_j_old*A[:,j]
                    # s_i_old = -s[i] (current s[i] is new)
                    # So As_new = As_old + 2*s[i]*A[:,i] + 2*s[j]*A[:,j]
                    
                    # Let's recompute As from scratch to be safe, n=100 is fast
                    As = adj @ s
                    
                    current_cut += total_gain
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                    improved = True
                    break # Restart pair search after a flip
            if time.time() > deadline - 5:
                break
    
    return best_labels, best_cut

def _simulated_annealing_polish(adj, labels, deadline, start_temp=1.0, cool_rate=0.995, min_temp=0.01):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    current_cut = 0.5 * (np.sum(adj) - s @ adj @ s)
    best_cut = current_cut
    best_labels = labels.copy()
    
    temp = start_temp
    rng = np.random.RandomState(123)
    
    while temp > min_temp and time.time() < deadline:
        v = rng.randint(n)
        
        old_s_v = s[v]
        As_v = adj[v] @ s
        gain = old_s_v * As_v
        
        new_cut = current_cut + gain
        
        if new_cut >= current_cut or rng.random() < np.exp((new_cut - current_cut) / temp):
            labels[v] = 1 - labels[v]
            s[v] = -s[v]
            current_cut = new_cut
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        temp *= cool_rate
    
    return best_labels, best_cut

def partition(adjacency):
    """
    Find a high-quality cut using BCD-SDP initialization, spectral initialization,
    greedy local search, pair-flip search, and simulated annealing.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    best_labels = None
    best_cut = -1
    
    # 1. BCD-SDP initialization + greedy refine
    if time.time() < deadline - 30.0:
        sdp_labels = _sdp_bcd_init(adj, n, deadline)
        refined_labels, cut_val = _greedy_refine(adj, sdp_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
            
        # Pair flip refine on the SDP result
        if time.time() < deadline - 20.0:
            pair_labels, pair_cut = _pair_flip_refine(adj, best_labels, deadline)
            if pair_cut > best_cut:
                best_cut = pair_cut
                best_labels = pair_labels.copy()

    # 2. Spectral initialization + greedy refine
    if time.time() < deadline - 20.0:
        spec_labels = _spectral_init(adj, n)
        refined_labels, cut_val = _greedy_refine(adj, spec_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
            
        # Pair flip refine on spectral result
        if time.time() < deadline - 15.0:
            pair_labels, pair_cut = _pair_flip_refine(adj, best_labels, deadline)
            if pair_cut > best_cut:
                best_cut = pair_cut
                best_labels = pair_labels.copy()

    # 3. Simulated annealing polish
    if best_labels is not None and time.time() < deadline - 10.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
            
        # Pair flip after SA
        if time.time() < deadline - 5.0:
            pair_labels, pair_cut = _pair_flip_refine(adj, best_labels, deadline)
            if pair_cut > best_cut:
                best_cut = pair_cut
                best_labels = pair_labels.copy()

    # 4. Random restarts with greedy refine + pair flip
    rng = np.random.RandomState(42)
    restart_count = 0
    
    while time.time() < deadline - 2.0:
        random_labels = np.random.randint(0, 2, n).astype(np.float64)
        refined_labels, cut_val = _greedy_refine(adj, random_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
            
        # Occasionally do pair flip on the best
        if restart_count % 5 == 0 and time.time() < deadline - 5.0:
            pair_labels, pair_cut = _pair_flip_refine(adj, best_labels, deadline)
            if pair_cut > best_cut:
                best_cut = pair_cut
                best_labels = pair_labels.copy()
                
        restart_count += 1
        if restart_count > 2000:
            break

    # 5. Final polish
    if best_labels is not None and time.time() < deadline - 1.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
            
        pair_labels, pair_cut = _pair_flip_refine(adj, best_labels, deadline)
        if pair_cut > best_cut:
            best_cut = pair_cut
            best_labels = pair_labels.copy()

    if best_labels is None:
        best_labels = np.random.randint(0, 2, n).astype(np.float64)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
