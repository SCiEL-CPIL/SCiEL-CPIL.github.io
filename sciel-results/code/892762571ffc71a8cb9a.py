# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    s = 2 * labels.astype(np.float64) - 1  # ±1
    E_total = 0.5 * adj.sum()
    s_As = s @ (adj @ s)
    cut = E_total - 0.25 * s_As
    return int(round(cut))


def _greedy_refine(adj, labels, max_passes=100, deadline=None):
    """
    Flip-based greedy local search with incremental updates.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2 * labels.astype(np.float64) - 1
    
    # Compute (A s)
    As = adj @ s
    
    # gain[v] = 0.5 * s[v] * As[v]
    gain = 0.5 * s * As
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for v in range(n):
            if gain[v] > 1e-9:
                # Flip v
                s[v] = -s[v]
                labels[v] = 1 - labels[v]
                improved = True
                # Update As for all neighbors of v
                delta = 2.0 * s[v]
                As += delta * adj[v]
                # Update gain for v and its neighbors
                gain[v] = 0.5 * s[v] * As[v]
                neighbors = np.nonzero(adj[v])[0]
                if len(neighbors) > 0:
                    gain[neighbors] = 0.5 * s[neighbors] * As[neighbors]
        if not improved:
            break
    
    return labels


def _simulated_annealing(adj, labels, deadline, rng, T0=5.0, cooling=0.95, min_T=0.1):
    """
    Simulated annealing to escape local optima.
    Returns improved labels.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2 * labels.astype(np.float64) - 1
    As = adj @ s
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T = T0
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Randomly pick a vertex to flip
        v = rng.integers(0, n)
        
        # Compute gain of flipping v
        # gain = 0.5 * s[v] * As[v]
        gain_v = 0.5 * s[v] * As[v]
        
        # Accept with probability 1 if gain > 0, else exp(gain/T)
        if gain_v > 0 or (T > 0 and rng.random() < np.exp(gain_v / T)):
            # Flip v
            s[v] = -s[v]
            labels[v] = 1 - labels[v]
            current_cut += gain_v
            
            # Update As
            delta = 2.0 * s[v]
            As += delta * adj[v]
            
            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= cooling
    
    return best_labels


def _spectral_init(adj, n):
    """
    Use the Fiedler vector (2nd smallest eigenvector of Laplacian) to initialize.
    """
    degree = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = sp.diags(degree)
    A_sparse = sp.csr_matrix(adj)
    L = D - A_sparse
    
    try:
        k = min(4, n - 1)
        if k < 2:
            return np.arange(n) % 2
        evals, evecs = spla.eigsh(L, k=k, which='SM')
        fiedler = evecs[:, 1]
        labels = (fiedler > 0).astype(int)
        return labels
    except Exception:
        return np.arange(n) % 2


def _sdp_relaxation(adj, n, max_iter=200, lr=0.05, deadline=None):
    """
    Goemans-Williamson SDP relaxation via projected gradient ascent.
    Solves: max trace(X * (D - A)) s.t. X_ii = 1, X >= 0
    where D is the degree matrix.
    Returns the optimal X matrix.
    """
    degree = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = np.diag(degree)
    L = D - adj  # Laplacian
    
    # Initialize X as identity (satisfies X_ii = 1)
    X = np.eye(n, dtype=np.float64)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Gradient of objective: -2 * (L * X) (since we're maximizing trace(X*L))
        # Actually, gradient of trace(X * L) w.r.t. X is L
        # So we want to maximize, so gradient ascent: X += lr * L
        # But we need to maintain symmetry and X_ii = 1
        
        grad = L  # gradient of trace(X * L)
        
        # Projected gradient step
        X_new = X + lr * grad
        
        # Symmetrize
        X_new = 0.5 * (X_new + X_new.T)
        
        # Enforce diagonal = 1
        np.fill_diagonal(X_new, 1.0)
        
        # Project onto PSD cone (via eigenvalue clipping)
        try:
            eigvals, eigvecs = np.linalg.eigh(X_new)
            eigvals = np.maximum(eigvals, 0)
            X_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-symmetrize and fix diagonal
            X_new = 0.5 * (X_new + X_new.T)
            np.fill_diagonal(X_new, 1.0)
        except Exception:
            pass
        
        X = X_new
    
    return X


def _hyperplane_rounding(X, rng, n_rounds=10):
    """
    Goemans-Williamson randomized hyperplane rounding.
    Given SDP solution X, generate diverse binary labels by
    sampling random hyperplanes and assigning based on sign of v^T * x_i.
    Returns list of label arrays.
    """
    n = X.shape[0]
    labels_list = []
    
    for _ in range(n_rounds):
        # Sample random vector
        v = rng.standard_normal(n)
        
        # Compute projections
        projections = X @ v
        
        # Assign labels based on sign
        labels = (projections > 0).astype(int)
        
        # Handle case where all labels are the same
        if labels.sum() == 0 or labels.sum() == n:
            # Flip half randomly
            half = n // 2
            indices = rng.permutation(n)
            labels[indices[:half]] = 1
        
        labels_list.append(labels)
    
    return labels_list


def _pairwise_swap_search(adj, labels, deadline=None):
    """
    Local search that considers swapping pairs of vertices.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2 * labels.astype(np.float64) - 1
    As = adj @ s
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for _pass in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try swapping pairs of vertices with different labels
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j]:
                    # Compute gain of swapping i and j
                    # This is complex, so we'll just try the swap and check
                    s[i], s[j] = s[j], s[i]
                    labels[i], labels[j] = labels[j], labels[i]
                    
                    # Recompute cut
                    new_cut = _cut_size(adj, labels)
                    
                    if new_cut > best_cut:
                        best_cut = new_cut
                        best_labels = labels.copy()
                        improved = True
                    else:
                        # Revert
                        s[i], s[j] = s[j], s[i]
                        labels[i], labels[j] = labels[j], labels[i]
        
        if not improved:
            break
    
    return best_labels


def partition(adjacency):
    """
    Maximum Cut via SDP Relaxation + Hyperplane Rounding + Greedy + Simulated Annealing.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    deadline = time.time() + 140  # safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = 0.5 * (adj + adj.T)
    
    best_labels = None
    best_cut = -1
    
    # 1. SDP relaxation + hyperplane rounding
    if time.time() < deadline - 30:
        X = _sdp_relaxation(adj, n, max_iter=150, lr=0.03, deadline=deadline - 30)
        sdp_labels_list = _hyperplane_rounding(X, rng, n_rounds=15)
        
        for labels_sdp in sdp_labels_list:
            if time.time() > deadline - 10:
                break
            refined = _greedy_refine(adj, labels_sdp, max_passes=60, deadline=deadline - 10)
            cut = _cut_size(adj, refined)
            if cut > best_cut:
                best_cut = cut
                best_labels = refined
    
    # 2. Spectral initialization as fallback/backup
    if best_labels is None or time.time() < deadline - 20:
        labels_spec = _spectral_init(adj, n)
        if time.time() < deadline - 10:
            refined_spec = _greedy_refine(adj, labels_spec, max_passes=60, deadline=deadline - 10)
            cut_spec = _cut_size(adj, refined_spec)
            if cut_spec > best_cut:
                best_cut = cut_spec
                best_labels = refined_spec
    
    # 3. Random restarts
    if best_labels is None:
        best_labels = rng.integers(0, 2, size=n)
        best_cut = _cut_size(adj, best_labels)
    
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        labels_r = rng.integers(0, 2, size=n)
        labels_r = _greedy_refine(adj, labels_r, max_passes=40, deadline=deadline - 15)
        cut_r = _cut_size(adj, labels_r)
        if cut_r > best_cut:
            best_cut = cut_r
            best_labels = labels_r
    
    # 4. Simulated annealing on the best solution
    if time.time() < deadline - 5:
        sa_deadline = deadline - 3
        sa_labels = _simulated_annealing(adj, best_labels, sa_deadline, rng, T0=5.0, cooling=0.95, min_T=0.1)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # 5. Pairwise swap search
    if time.time() < deadline - 2:
        swap_labels = _pairwise_swap_search(adj, best_labels, deadline=deadline - 1)
        swap_cut = _cut_size(adj, swap_labels)
        if swap_cut > best_cut:
            best_cut = swap_cut
            best_labels = swap_labels
    
    # 6. Final polish
    if time.time() < deadline:
        polished = _greedy_refine(adj, best_labels, max_passes=80, deadline=deadline)
        polished_cut = _cut_size(adj, polished)
        if polished_cut > best_cut:
            best_labels = polished
            best_cut = polished_cut
    
    return best_labels
# EVOLVE-BLOCK-END
