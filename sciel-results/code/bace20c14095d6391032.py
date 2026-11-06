# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    # labels is 0/1 array. Convert to ±1.
    s = 2.0 * labels.astype(np.float64) - 1.0
    # s^T A s = sum_{v,u} s_v A_vu s_u
    # For an edge (v,u) where v != u: contributes 2 * s_v * s_u
    # If same side: s_v * s_u = 1, contributes 2
    # If different side: s_v * s_u = -1, contributes -2
    # Total edges E = sum of all entries / 2 (since symmetric, no self-loops assumed)
    # s^T A s = 2 * (same_side_edges - different_side_edges)
    # Let C = cut size (different_side_edges), S = same_side_edges
    # E = C + S
    # s^T A s = 2*(S - C) = 2*(E - C - C) = 2*E - 4*C
    # So C = (2*E - s^T A s) / 4
    E = 0.5 * adj.sum()
    val = s @ (adj @ s)
    return (2.0 * E - val) / 4.0


def _greedy_refine(adj, labels, max_sweeps=100, deadline=None):
    """Greedy local improvement. Returns improved labels."""
    n = len(adj)
    labels = labels.copy()
    adj_bool = adj.astype(bool)
    
    # Precompute degrees for faster computation? No, we need dynamic neighbor sums.
    # For n=100, n^2 is 10000, which is very fast.
    
    for _ in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for v in range(n):
            # Count neighbors in same set vs different set
            # labels == labels[v] gives mask
            # adj_bool[v] is the row of neighbors
            # We want sum of adj_bool[v, labels == labels[v]]
            # This is a bit slow in pure python loops if done naively, but n=100 is tiny.
            # Using numpy:
            same_mask = (labels == labels[v])
            diff_mask = ~same_mask
            
            # Number of neighbors in same set
            n_same = adj_bool[v, same_mask].sum()
            # Number of neighbors in diff set
            n_diff = adj_bool[v, diff_mask].sum()
            
            if n_same > n_diff:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
            
    return labels


def _spectral_init(adj):
    """Initialize labels using the Fiedler vector (second eigenvector of Laplacian)."""
    n = len(adj)
    adj_float = adj.astype(np.float64)
    
    # Build Laplacian L = D - A
    # D is diagonal matrix of degrees
    degrees = adj_float.sum(axis=1)
    D = sp.diags(degrees, 0)
    L = D - sp.csr_matrix(adj_float)
    
    # We want the second smallest eigenvalue/eigenvector of L.
    # Use scipy.sparse.linalg.eigsh.
    # k=2 to get the two smallest.
    # sigma=0 for smallest magnitude, or use shift-invert.
    # For n=100, dense eigen is also fine and simpler, but let's try sparse.
    try:
        # eigsh(L, k=2, sigma=0, which='LM')
        # sigma=0 with which='LM' finds eigenvalues closest to 0.
        # Since L is positive semi-definite, the smallest eigenvalue is 0 (constant vector).
        # The second smallest is the Fiedler value.
        # However, eigsh with sigma=0 can be tricky if 0 is an eigenvalue.
        # Alternative: use dense np.linalg.eigh for n=100. It's very fast.
        L_dense = L.toarray()
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        
        # eigvals are sorted in ascending order.
        # eigvecs[:, 0] corresponds to smallest eigenvalue (should be ~0, constant vector).
        # eigvecs[:, 1] corresponds to second smallest (Fiedler vector).
        fiedler = eigvecs[:, 1]
        
        # Threshold at 0 to get 0/1 labels.
        # Mean centering is not strictly necessary for thresholding at 0 if the vector is centered,
        # but Fiedler vector is orthogonal to 1, so mean is 0.
        labels = (fiedler > 0).astype(int)
        
        # Ensure both classes are non-empty (should be, but just in case)
        if labels.sum() == 0 or labels.sum() == n:
            # Fallback to random or alternating
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2


def _random_init(n, rng):
    """Random initialization."""
    return rng.integers(0, 2, size=n).astype(int)


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Deadline: 150s limit, leave 10s buffer.
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Convert to sparse matrix for efficiency in spectral if needed, 
    # but for n=100 dense is fine.
    # Let's stick to dense for simplicity and speed in small n.
    adj_dense = np.asarray(adjacency, dtype=np.float64)
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Init
    labels_spec = _spectral_init(adj_dense)
    labels_refined = _greedy_refine(adj_dense, labels_spec, max_sweeps=200, deadline=deadline)
    cut_val = _cut_size(adj_dense, labels_refined)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = labels_refined.copy()
    
    # 2. Multiple Random Restarts
    # How many restarts can we afford?
    # Each restart: init (fast) + greedy (fast).
    # Let's try to use remaining time for random restarts.
    rng = np.random.default_rng(42)
    
    # Estimate time per iteration
    # We'll just loop until deadline.
    while time.time() < deadline - 5.0:
        labels_rand = _random_init(n, rng)
        labels_refined = _greedy_refine(adj_dense, labels_rand, max_sweeps=100, deadline=deadline)
        cut_val = _cut_size(adj_dense, labels_refined)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels_refined.copy()
            
        # Optional: Try a few "perturbed" spectral starts?
        # Or just keep doing random restarts.
        # Random restarts with greedy local search is a standard heuristic for MaxCut.
        
        # Check time more frequently
        if time.time() > deadline:
            break

    return best_labels
# EVOLVE-BLOCK-END
