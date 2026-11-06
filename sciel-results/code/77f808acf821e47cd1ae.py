# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1  # ±1
    # s^T A s counts 2 for each same-side edge and -2 for each crossing edge? No.
    # Let's use the direct formula:
    # cut = sum_{i,j} A[i,j] * (labels[i] != labels[j])
    # In terms of s: labels[i] != labels[j] iff s[i]*s[j] == -1
    # (1 - s[i]*s[j])/2 is 1 if different, 0 if same
    # cut = 0.5 * sum_{i,j} A[i,j] * (1 - s[i]*s[j])
    #     = 0.5 * (sum A - s^T A s)
    # But sum A is 2*E (since undirected, A is symmetric with 0 diagonal)
    # Actually, let's just compute it directly to be safe and fast with numpy
    # For 100 nodes, O(n^2) is 10,000 ops, which is trivial.
    
    # Efficient vectorized way:
    # diff = labels[:, None] != labels[None, :]
    # return np.sum(adj * diff)
    
    # Even faster using the s formulation:
    # s^T A s = sum_i sum_j s_i A_ij s_j
    # For undirected graph with A symmetric:
    # s^T A s = sum_{i<j} 2 * A_ij * s_i s_j
    # If s_i s_j = 1 (same), contributes 2 A_ij
    # If s_i s_j = -1 (diff), contributes -2 A_ij
    # Let S = sum of same-side edges (counted once), D = sum of diff-side edges (counted once)
    # s^T A s = 2S - 2D
    # Also 2E = 2S + 2D => S + D = E
    # So 2S - 2D = 2S - 2(E - S) = 4S - 2E
    # => S = (s^T A s + 2E) / 4
    # D = E - S = E - (s^T A s + 2E)/4 = (4E - s^T A s - 2E)/4 = (2E - s^T A s)/4
    # Cut size is D.
    
    # Let's just do the direct boolean mask approach, it's clear and fast enough for n=100
    diff = (labels[:, None] != labels[None, :])
    return np.sum(adj * diff)


def _spectral_init(adj, n):
    """Generate initial labels using spectral method (Fiedler vector)."""
    # Build Laplacian L = D - A
    # For undirected graph
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    # Find the second smallest eigenvector of L
    # Use scipy sparse eigensolver for efficiency
    L_sparse = sp.csr_matrix(L)
    
    try:
        # We want the Fiedler vector (eigenvector corresponding to 2nd smallest eigenvalue)
        # sigma='LM' finds largest magnitude. We can shift.
        # Or just use dense eig for n=100, it's fast.
        evals, evecs = np.linalg.eigh(L)
        # evecs[:, 1] is the Fiedler vector
        fiedler = evecs[:, 1]
    except:
        # Fallback to random
        return np.random.randint(0, 2, n).astype(np.float64)
    
    # Use sign of Fiedler vector as initial labels
    labels = (fiedler > 0).astype(np.float64)
    
    # Ensure both classes are non-empty
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n).astype(np.float64)
    
    return labels


def _greedy_refine(adj, labels, deadline):
    """
    Vectorized best-improvement local search with incremental updates.
    Repeatedly find the vertex whose flip increases the cut the most.
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
        
        # Update As incrementally: As[u] changes by -2*A[u,v]*old_s_v = +2*A[u,v]*new_s_v
        As += 2 * adj[:, v] * s[v]
        gains = s * As
    
    return labels, 0.5 * (np.sum(adj) - s @ adj @ s)


def _random_init(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, n).astype(np.float64)


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
    Find a high-quality cut using spectral initialization, greedy local search,
    and simulated annealing polish with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral initialization + greedy refine
    spec_labels = _spectral_init(adj, n)
    refined_labels, cut_val = _greedy_refine(adj, spec_labels, deadline)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # 2. Simulated annealing polish on the best solution
    if time.time() < deadline - 5.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # 3. Random restarts with greedy refine + occasional SA polish
    rng = np.random.RandomState(42)
    restart_count = 0
    sa_every = 10
    
    while time.time() < deadline - 2.0:
        random_labels = _random_init(n)
        refined_labels, cut_val = _greedy_refine(adj, random_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        restart_count += 1
        
        if restart_count % sa_every == 0 and time.time() < deadline - 5.0:
            sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        if restart_count > 2000:
            break
    
    # 4. Final SA polish if time permits
    if time.time() < deadline - 1.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = _random_init(n)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
