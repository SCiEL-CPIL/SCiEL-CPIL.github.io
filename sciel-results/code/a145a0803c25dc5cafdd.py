# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    n = len(labels)
    s = 2.0 * labels.astype(np.float64) - 1.0  # ±1
    # cut = 0.5 * (sum(A) - s^T A s)
    # But for undirected graph with A symmetric, sum(A) = 2*E
    # Actually: cut = sum_{i<j} A[i,j] * (1 - s[i]*s[j]) / 2 ... no
    # Let's use: cut = 0.25 * (n^2 - s^T A s) ... no that's not right either.
    # Direct: cut = sum over i<j of A[i,j] * (labels[i] != labels[j])
    # In terms of s: labels[i] != labels[j] iff s[i]*s[j] == -1
    # So cut = 0.5 * sum_{i,j} A[i,j] * (1 - s[i]*s[j])
    #        = 0.5 * (sum_{i,j} A[i,j] - sum_{i,j} A[i,j]*s[i]*s[j])
    # sum_{i,j} A[i,j] = 2*E (since undirected, each edge counted twice)
    # sum_{i,j} A[i,j]*s[i]*s[j] = s^T A s
    total = adj.sum()
    val = s @ (adj @ s)
    cut = 0.5 * (total - val)
    return int(round(cut))


def _spectral_init(adj):
    """Use the Fiedler vector (second smallest eigenvector of Laplacian) to initialize."""
    n = len(adj)
    # Laplacian L = D - A
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    # Get eigenvalues and eigenvectors
    # For n=100, full eigendecomposition is fine
    eigvals, eigvecs = np.linalg.eigh(L)
    # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    # eigvals are sorted in ascending order
    fiedler = eigvecs[:, 1]
    # Threshold at 0
    labels = (fiedler >= 0).astype(np.int64)
    return labels


def _greedy_refine(adj, labels, deadline=None):
    """Greedy local search: flip a vertex if it increases the cut size."""
    n = len(adj)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    # Precompute for speed
    # For each vertex v, gain from flipping = (sum of neighbors in other set) - (sum of neighbors in same set)
    # = 2 * (sum of neighbors in other set) - degree[v]
    # Actually: if we flip v, edges from v to same set become crossing, edges from v to other set become non-crossing
    # Net change = (same_count) - (other_count) where same_count = neighbors in same set, other_count = neighbors in other set
    # Wait: if v is in set 0, and we flip to set 1:
    #   Edges from v to set 0 (same) were not crossing, now become crossing: +same_count
    #   Edges from v to set 1 (other) were crossing, now become not crossing: -other_count
    #   Net = same_count - other_count
    # We flip if net > 0, i.e., same_count > other_count
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if deadline is not None and time.time() > deadline:
                break
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
        if deadline is not None and time.time() > deadline:
            break
    
    return labels, _cut_size(adj, labels)


def _random_init(n, seed=None):
    """Random 50/50 initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.randint(0, 2, size=n).astype(np.int64)


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline: 120s wall clock, leave 10s margin
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # Strategy: multiple restarts with different initializations
    # 1. Spectral initialization
    # 2. Random initializations
    # 3. Alternating initialization
    
    # First try spectral
    labels_spec = _spectral_init(adj)
    labels_ref, cut = _greedy_refine(adj, labels_spec, deadline)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_ref.copy()
    
    # Try alternating
    labels_alt = np.arange(n) % 2
    labels_ref, cut = _greedy_refine(adj, labels_alt, deadline)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_ref.copy()
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    num_random = 50
    for i in range(num_random):
        if time.time() > deadline:
            break
        labels_rand = _random_init(n, seed=rng.randint(0, 2**31 - 1))
        labels_ref, cut = _greedy_refine(adj, labels_rand, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_ref.copy()
    
    # If we have time, do more sophisticated search
    # Try a simple simulated annealing or just more greedy passes with perturbations
    if best_labels is not None and time.time() < deadline - 5.0:
        # Perturbation + re-refine
        for _ in range(20):
            if time.time() > deadline:
                break
            perturbed = best_labels.copy()
            # Flip a random subset of vertices
            num_flip = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flip, replace=False)
            perturbed[flip_idx] = 1 - perturbed[flip_idx]
            labels_ref, cut = _greedy_refine(adj, perturbed, deadline)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels_ref.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.zeros(n, dtype=np.int64)
    
    return best_labels
# EVOLVE-BLOCK-END
