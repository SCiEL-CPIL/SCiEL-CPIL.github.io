"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, max_time=None, start_time=None):
    """Greedy local improvement: flip vertices that increase cut size.
    
    Uses efficient gain computation.
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    labels = labels.copy()
    
    if start_time is None:
        start_time = time.time()
    
    # Precompute for efficiency
    # gain[v] = (sum of edges from v to same side) - (sum of edges from v to opposite side)
    # If gain[v] > 0, flipping v increases the cut
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if max_time is not None and time.time() - start_time > max_time:
                return labels
            
            # Count edges to same side and opposite side
            same_side = labels == labels[v]
            opposite_side = labels != labels[v]
            
            edges_same = A[v, same_side].sum()
            edges_opposite = A[v, opposite_side].sum()
            
            gain = edges_same - edges_opposite
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _spectral_partition(adjacency, rng):
    """Generate a partition using spectral methods (Fiedler vector)."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    # Compute Laplacian
    degree = A.sum(axis=1)
    L = np.diag(degree) - A
    
    # Find Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
    # Use power iteration on shifted Laplacian or just eigendecomposition for n=100
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Second smallest eigenvalue's eigenvector
        fiedler = eigvecs[:, 1]
    except Exception:
        fiedler = rng.normal(size=n)
    
    # Partition based on sign of Fiedler vector
    labels = (fiedler > 0).astype(np.int8)
    
    # Ensure both partitions are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        # Fall back to random
        labels = rng.integers(0, 2, size=n).astype(np.int8)
    
    return labels


def _random_hyperplane_cut(adjacency, rng, n_vectors=10):
    """Generate partitions using random hyperplane cuts from eigenvectors."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    # Compute Laplacian
    degree = A.sum(axis=1)
    L = np.diag(degree) - A
    
    # Get eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        eigvecs = rng.normal(size=(n, n))
    
    # Use the first few eigenvectors (lowest eigenvalues) for the embedding
    # The Fiedler vector is the most useful for spectral partitioning
    # But we can also use random linear combinations
    
    best_labels = None
    best_cut = -1
    
    for i in range(n_vectors):
        # Random linear combination of the first few eigenvectors
        weights = rng.normal(size=min(5, n))
        combined = eigvecs[:, :len(weights)] @ weights
        
        # Partition based on sign
        labels = (combined > 0).astype(np.int8)
        
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            continue
        
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    if best_labels is None:
        best_labels = rng.integers(0, 2, size=n).astype(np.int8)
    
    return best_labels


def partition(adjacency):
    """Partition vertices into two sets to maximize the cut size.
    
    Uses spectral initialization followed by local search, with multiple restarts.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral partitioning with local search
    spectral_labels = _spectral_partition(adjacency, rng)
    improved = _local_search(adjacency, spectral_labels, max_time=30, start_time=start_time)
    cut = _cut_size(adjacency, improved)
    if cut > best_cut:
        best_cut = cut
        best_labels = improved.copy()
    
    if time.time() > deadline:
        return best_labels
    
    # Strategy 2: Random hyperplane cuts from eigenvectors
    hp_labels = _random_hyperplane_cut(adjacency, rng, n_vectors=20)
    improved = _local_search(adjacency, hp_labels, max_time=30, start_time=start_time)
    cut = _cut_size(adjacency, improved)
    if cut > best_cut:
        best_cut = cut
        best_labels = improved.copy()
    
    if time.time() > deadline:
        return best_labels
    
    # Strategy 3: Multiple random restarts with local search
    for restart in range(20):
        if time.time() > deadline:
            break
        
        random_labels = rng.integers(0, 2, size=n).astype(np.int8)
        improved = _local_search(adjacency, random_labels, max_time=10, start_time=start_time)
        cut = _cut_size(adjacency, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    # Strategy 4: Alternating start (original) with more iterations
    if time.time() < deadline:
        alt_labels = np.arange(n) % 2
        improved = _local_search(adjacency, alt_labels, max_time=20, start_time=start_time)
        cut = _cut_size(adjacency, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    # Final polish: one more local search pass on the best solution
    if time.time() < deadline and best_labels is not None:
        final = _local_search(adjacency, best_labels, max_time=15, start_time=start_time)
        final_cut = _cut_size(adjacency, final)
        if final_cut > best_cut:
            best_labels = final
    
    return best_labels
# EVOLVE-BLOCK-END
