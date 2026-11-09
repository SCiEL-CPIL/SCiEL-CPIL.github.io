"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # Edge crosses if labels differ
    # We can compute: for each edge (i,j) with i<j, check if labels[i] != labels[j]
    # Using matrix: cut = 0.5 * sum over i,j of A[i,j] * (labels[i] != labels[j])
    # More efficiently:
    # For vertex v in set 0, count edges to set 1
    # For vertex v in set 1, count edges to set 0
    # Total cut = sum_{v in 0} deg_to_1(v) + sum_{v in 1} deg_to_0(v)
    # But each edge is counted once, so this is the cut size.
    
    # Vectorized: 
    # Let's use: cut = 0.5 * (sum of A * (1 - labels_i * labels_j mapped))
    # Actually simpler: 
    # diff = (labels[:, None] != labels[None, :])  # n x n boolean
    # cut = (adjacency * diff).sum() // 2  # since each edge counted twice
    # But for n=100 this is fine.
    
    # Even simpler with numpy:
    # cut = np.sum(adjacency[labels[:, None] != labels[None, :]]) // 2
    # But let's do it more efficiently:
    
    # For each vertex, count neighbors in opposite set
    # labels is 0/1 array
    # opposite = 1 - labels
    # For vertex v, count = adjacency[v, labels == opposite[v]].sum()
    # Total cut = sum of these / 2? No, each edge is counted exactly once this way
    # because if v is in set 0 and u is in set 1, we count it when looking at v.
    # If v is in set 1 and u is in set 0, we count it when looking at v.
    # So total is exactly the cut size.
    
    opposite = 1 - labels
    # For each vertex v, count neighbors in opposite set
    # adjacency[v, opposite[v] == labels] gives neighbors in opposite set
    # We can do: 
    # counts = np.array([adjacency[v, labels == opposite[v]].sum() for v in range(n)])
    # cut = counts.sum()
    
    # Vectorized approach:
    # Create mask: for each row v, we want columns where labels == opposite[v]
    # This is: labels[None, :] == opposite[:, None]  # n x n
    mask = labels[None, :] == opposite[:, None]  # n x n, True where neighbor is in opposite set
    counts = (adjacency * mask).sum(axis=1)
    cut = counts.sum()
    return cut


def _local_improvement(adjacency, labels, max_iterations=100, time_limit=None, start_time=None):
    """Greedy local improvement. Returns improved labels and cut size."""
    n = len(labels)
    labels = labels.copy()
    
    if time_limit is not None and start_time is None:
        start_time = time.time()
    
    for iteration in range(max_iterations):
        if time_limit is not None and (time.time() - start_time) > time_limit - 5:
            break
        
        improved = False
        for v in range(n):
            if time_limit is not None and (time.time() - start_time) > time_limit - 5:
                break
            
            current_label = labels[v]
            same_count = adjacency[v, labels == current_label].sum()
            other_count = adjacency[v, labels != current_label].sum()
            
            if other_count > same_count:
                labels[v] = 1 - current_label
                improved = True
        
        if not improved:
            break
    
    return labels


def _spectral_init(adjacency):
    """Initialize using the Fiedler vector of the graph Laplacian."""
    n = len(adjacency)
    adj_float = adjacency.astype(float)
    degrees = adj_float.sum(axis=1)
    
    # Build Laplacian L = D - A
    L = np.diag(degrees) - adj_float
    
    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # For connected graphs, the smallest eigenvalue is 0 with eigenvector all ones
        # The second smallest gives a good cut
        fiedler = eigenvectors[:, 1]
    except Exception:
        # Fallback to random
        return np.random.randint(0, 2, size=n)
    
    # Split by sign
    labels = (fiedler >= 0).astype(np.int8)
    
    # Ensure both sets are non-empty (if all same, flip one)
    if labels.all() or not labels.any():
        labels[0] = 1 - labels[0]
    
    return labels


def partition(adjacency):
    """
    Find a good partition for Maximum Cut.
    
    Args:
        adjacency: bool matrix (n x n) where adjacency[i,j] is True if edge (i,j) exists
    
    Returns:
        0/1 label per vertex (numpy array of length n)
    """
    start_time = time.time()
    time_limit = 150  # seconds, with safety margin
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    n = len(adjacency)
    
    if n == 0:
        return np.array([], dtype=np.int8)
    
    if n == 1:
        return np.array([0], dtype=np.int8)
    
    # Try multiple strategies and keep the best
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + local improvement
    labels_spec = _spectral_init(adjacency)
    labels_spec = _local_improvement(adjacency, labels_spec, max_iterations=50, 
                                      time_limit=time_limit, start_time=start_time)
    cut_spec = _cut_size(adjacency, labels_spec)
    if cut_spec > best_cut:
        best_cut = cut_spec
        best_labels = labels_spec.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - 30:
        return best_labels
    
    # Strategy 2: Random initialization + local improvement (multiple restarts)
    num_restarts = 5
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
        
        labels_rand = np.random.randint(0, 2, size=n)
        labels_rand = _local_improvement(adjacency, labels_rand, max_iterations=50,
                                          time_limit=time_limit, start_time=start_time)
        cut_rand = _cut_size(adjacency, labels_rand)
        if cut_rand > best_cut:
            best_cut = cut_rand
            best_labels = labels_rand.copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - 15:
        return best_labels
    
    # Strategy 3: Alternating initialization + local improvement (original)
    labels_alt = np.arange(n) % 2
    labels_alt = _local_improvement(adjacency, labels_alt, max_iterations=100,
                                     time_limit=time_limit, start_time=start_time)
    cut_alt = _cut_size(adjacency, labels_alt)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()
    
    # If we have time, do additional random restarts
    elapsed = time.time() - start_time
    if elapsed < time_limit - 20:
        while time.time() - start_time < time_limit - 10:
            labels_rand = np.random.randint(0, 2, size=n)
            labels_rand = _local_improvement(adjacency, labels_rand, max_iterations=30,
                                              time_limit=time_limit, start_time=start_time)
            cut_rand = _cut_size(adjacency, labels_rand)
            if cut_rand > best_cut:
                best_cut = cut_rand
                best_labels = labels_rand.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
