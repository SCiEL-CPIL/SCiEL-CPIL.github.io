"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(A, labels):
    """Compute the number of edges crossing the partition."""
    # A is bool matrix, labels is 0/1 array
    # Edge (i,j) crosses if A[i,j] and labels[i] != labels[j]
    # Since A is symmetric (undirected), we count each edge once
    diff = (labels[:, None] != labels[None, :])
    return int((A & diff).sum() // 2)


def _local_search(A, labels, max_iters=20):
    """Greedy local improvement: flip vertices that increase cut size.
    
    For vertex v, if we flip it:
    - Edges from v to same-side neighbors become crossing (gain)
    - Edges from v to opposite-side neighbors stop crossing (loss)
    Net gain = (same_side_degree) - (opposite_side_degree)
              = 2 * same_side_degree - total_degree[v]
    """
    n = len(labels)
    A_f = A.astype(np.float64)
    labels = labels.copy()
    degree = A_f.sum(axis=1)
    
    for _ in range(max_iters):
        improved = False
        for v in range(n):
            # Count neighbors in same set vs opposite set
            same_mask = (labels == labels[v])
            same_deg = A_f[v, same_mask].sum()
            opp_deg = degree[v] - same_deg
            gain = same_deg - opp_deg
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    return labels


def _spectral_partition(A):
    """Use spectral method to get initial partition.
    
    Compute the second eigenvector of the graph Laplacian,
    then partition based on sign.
    """
    n = len(A)
    A_f = A.astype(np.float64)
    degree = A_f.sum(axis=1)
    D = np.diag(degree)
    L = D - A_f
    
    # Use scipy for eigenvalue computation
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        L_sparse = csr_matrix(L)
        # Find the smallest non-zero eigenvalue (Fiedler vector)
        # eigsh finds eigenvalues closest to sigma
        # We want the second smallest eigenvalue (index 1)
        # sigma=0 finds eigenvalues closest to 0
        # We need to exclude the trivial eigenvector (constant)
        try:
            # Find k=2 smallest eigenvalues
            vals, vecs = eigsh(L_sparse, k=2, sigma=0.0, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            vecs = vecs[:, idx]
            # The Fiedler vector is the second one
            fiedler = vecs[:, 1]
        except Exception:
            # Fallback: use numpy
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            vecs = vecs[:, idx]
            fiedler = vecs[:, 1]
        
        # Partition based on sign of Fiedler vector
        labels = (fiedler > 0).astype(np.int32)
        # Ensure both classes are non-empty
        if labels.all() or not labels.any():
            labels = np.arange(n) % 2
        return labels
    except Exception:
        # Fallback to random
        return np.random.randint(0, 2, n)


def partition(adjacency):
    """Find a good partition of the graph to maximize cut size.
    
    Strategy:
    1. Use spectral method for initial partition
    2. Run local search to improve
    3. Try multiple random restarts and keep the best
    """
    n = len(adjacency)
    A = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_labels = None
    best_cut = -1
    
    # First try spectral initialization
    try:
        spec_labels = _spectral_partition(A)
        improved = _local_search(A, spec_labels, max_iters=30)
        cut = _cut_size(A, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    except Exception:
        pass
    
    # Try random restarts
    np.random.seed(42)
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initial partition
        labels = np.random.randint(0, 2, n)
        improved = _local_search(A, labels, max_iters=30)
        cut = _cut_size(A, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    # Additional local search passes on the best solution
    if best_labels is not None:
        if time.time() < deadline:
            improved = _local_search(A, best_labels, max_iters=50)
            cut = _cut_size(A, improved)
            if cut > best_cut:
                best_cut = cut
                best_labels = improved.copy()
        
        # Try more aggressive local search with multiple passes
        if time.time() < deadline:
            for _ in range(5):
                if time.time() > deadline:
                    break
                improved = _local_search(A, best_labels, max_iters=20)
                cut = _cut_size(A, improved)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = improved.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
