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


def _greedy_refine(adj, labels, max_passes=80, deadline=None):
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


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Greedy Local Search + Simulated Annealing.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    adj = 0.5 * (adj + adj.T)
    
    # Spectral initialization
    labels = _spectral_init(adj, n)
    labels = _greedy_refine(adj, labels, max_passes=80, deadline=deadline)
    best_labels = labels.copy()
    best_cut = _cut_size(adj, best_labels)
    
    # Multi-start with random inits
    rng = np.random.default_rng(42)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        labels_r = rng.integers(0, 2, size=n)
        labels_r = _greedy_refine(adj, labels_r, max_passes=50, deadline=deadline)
        cut_r = _cut_size(adj, labels_r)
        if cut_r > best_cut:
            best_cut = cut_r
            best_labels = labels_r
    
    # Simulated annealing on the best solution
    if time.time() < deadline:
        sa_deadline = deadline - 5  # leave some time
        sa_labels = _simulated_annealing(adj, best_labels, sa_deadline, rng, T0=5.0, cooling=0.95, min_T=0.1)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final polish: one more greedy pass on the best
    if time.time() < deadline:
        polished = _greedy_refine(adj, best_labels, max_passes=100, deadline=deadline)
        polished_cut = _cut_size(adj, polished)
        if polished_cut > best_cut:
            best_labels = polished
            best_cut = polished_cut
    
    return best_labels
# EVOLVE-BLOCK-END
