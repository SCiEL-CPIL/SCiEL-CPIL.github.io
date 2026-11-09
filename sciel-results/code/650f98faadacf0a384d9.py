"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Use spectral method to get initial partition via Fiedler-like vector."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    # For max-cut, we want to maximize x^T A x s.t. x in {-1,1}^n
    # Relax to x^T A x s.t. x^T x = n, then round
    # Use the eigenvector of A with largest eigenvalue (or use -A for min-cut style)
    # Actually, for max-cut, the spectral relaxation uses the eigenvector of A
    # corresponding to the largest eigenvalue, but we need to be careful.
    # Let's use the approach: maximize x^T A x. The top eigenvector of A gives
    # a good initial solution for max-cut.
    
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        # The largest eigenvalue is the last one (eigh returns in ascending order)
        # Use the eigenvector for the largest eigenvalue
        v = eigvecs[:, -1]
        # Round: positive -> 1, negative -> 0
        labels = (v >= 0).astype(np.int8)
    except:
        # Fallback to alternating
        labels = np.arange(n) % 2
    
    return labels


def _local_search(adjacency, labels, max_iters=10):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    labels = labels.copy()
    
    for _ in range(max_iters):
        improved = False
        for v in range(n):
            # Count edges to same side and other side
            same = adjacency[v][labels == labels[v]].sum()
            other = adjacency[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    
    return labels


def _simulated_annealing(adjacency, labels, start_temp=1.0, end_temp=0.01, 
                          n_iter=200, seed=None):
    """Simulated annealing to escape local optima."""
    n = len(adjacency)
    labels = labels.copy()
    rng = np.random.default_rng(seed)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for i in range(n_iter):
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        new_cut = _cut_size(adjacency, labels)
        delta = new_cut - current_cut
        
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-8)):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label
    
    return best_labels


def partition(adjacency):
    """Main partition function with deadline guard."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    # Generate multiple initial solutions
    best_labels = None
    best_cut = -1
    
    # 1. Spectral initialization
    labels_spec = _spectral_partition(adjacency)
    labels_spec = _local_search(adjacency, labels_spec)
    cut_spec = _cut_size(adjacency, labels_spec)
    if cut_spec > best_cut:
        best_cut = cut_spec
        best_labels = labels_spec.copy()
    
    # 2. Alternating initialization
    if time.time() < deadline:
        labels_alt = np.arange(n) % 2
        labels_alt = _local_search(adjacency, labels_alt)
        cut_alt = _cut_size(adjacency, labels_alt)
        if cut_alt > best_cut:
            best_cut = cut_alt
            best_labels = labels_alt.copy()
    
    # 3. Random initializations
    rng = np.random.default_rng(42)
    for i in range(5):
        if time.time() >= deadline:
            break
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        labels_rand = _local_search(adjacency, labels_rand)
        cut_rand = _cut_size(adjacency, labels_rand)
        if cut_rand > best_cut:
            best_cut = cut_rand
            best_labels = labels_rand.copy()
    
    # 4. Simulated annealing from best solution
    if time.time() < deadline - 10:
        labels_sa = _simulated_annealing(adjacency, best_labels, 
                                          start_temp=1.0, end_temp=0.01,
                                          n_iter=100, seed=123)
        cut_sa = _cut_size(adjacency, labels_sa)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
    
    # 5. Final local search polish
    if time.time() < deadline - 5:
        best_labels = _local_search(adjacency, best_labels, max_iters=5)
    
    return best_labels
# EVOLVE-BLOCK-END
