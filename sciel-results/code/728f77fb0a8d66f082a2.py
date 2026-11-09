"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Build Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    L_sparse = sp.csr_matrix(L)
    
    # Spectral embedding: approximate Goemans-Williamson SDP solution
    k = min(30, n - 1)
    
    try:
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, which='SM', return_eigenvectors=True)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Remove trivial eigenvector (eigenvalue ~ 0)
        mask = eigvals > 1e-8
        if np.sum(mask) > 0:
            embedding = eigvecs[:, mask]
        else:
            embedding = eigvecs
    except Exception:
        labels = np.arange(n) % 2
        labels = _local_search(adj, labels, n, max_rounds=20)
        return labels.astype(int)
    
    # Normalize rows to unit norm (SDP constraint: ||V_i|| = 1)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms < 1e-10] = 1.0
    embedding = embedding / row_norms
    
    # Randomized hyperplane projection (Goemans-Williamson step)
    best_labels = None
    best_cut = -1
    
    n_trials = 100
    dim = embedding.shape[1]
    
    for trial in range(n_trials):
        # Random unit hyperplane
        u = np.random.randn(dim)
        norm = np.linalg.norm(u)
        if norm < 1e-10:
            continue
        u /= norm
        
        # Project and partition
        proj = embedding @ u
        labels = (proj >= 0).astype(int)
        
        # Local search refinement
        labels = _local_search(adj, labels, n, max_rounds=20)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_labels = _local_search(adj, best_labels, n, max_rounds=20)
    
    return best_labels.astype(int)


def _cut_size(adj, labels):
    cross = (labels[:, None] != labels[None, :])
    return (adj * cross).sum() / 2


def _local_search(adj, labels, n, max_rounds=20):
    labels = labels.copy()
    for _ in range(max_rounds):
        improved = False
        order = np.random.permutation(n)
        for v in order:
            same = adj[v, labels == labels[v]].sum()
            other = adj[v, labels != labels[v]].sum()
            if other > same:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    return labels
# EVOLVE-BLOCK-END
