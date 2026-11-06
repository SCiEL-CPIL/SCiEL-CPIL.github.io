import numpy as np

def _run_local_search(labels, A, deg):
    """
    Runs best-improvement greedy local search starting from initial labels.
    Returns the final labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(1000):
        # Calculate sum of labels of neighbors for each vertex
        neighbor_labels_sum = A @ labels
        
        # Calculate number of same-label neighbors for each vertex
        # If labels[i] == 0: same = deg[i] - neighbor_labels_sum[i]
        # If labels[i] == 1: same = neighbor_labels_sum[i]
        same_counts = np.where(labels == 0, deg - neighbor_labels_sum, neighbor_labels_sum)
        
        # Calculate gain in cut size if vertex i is flipped
        # Gain = 2 * Same - Deg
        gains = 2 * same_counts - deg
        
        # Find the vertex with the maximum gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        # If the best possible flip does not improve the cut, we are at a local optimum
        if max_gain <= 0:
            break
        
        # Flip the vertex
        labels[max_gain_idx] = 1 - labels[max_gain_idx]
        
    # Calculate final cut size
    cut_size = np.sum(A * (labels[:, None] != labels[None, :])) / 2.0
    
    return labels, cut_size

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    A = adjacency.astype(float)
    deg = A.sum(axis=1)
    
    # Spectral Initialization
    try:
        L = np.diag(deg) - A
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        median_val = np.median(fiedler)
        init_labels = (fiedler > median_val).astype(int)
        
        # Ensure non-trivial partition if possible
        if init_labels.sum() == 0 or init_labels.sum() == n:
             # Fallback: use mean or just random if vector is flat
             # For connected graphs, Fiedler is not flat.
             # If flat, graph is disconnected?
             # Just use random or alternating
             init_labels = np.random.randint(0, 2, n)
    except Exception:
        init_labels = np.random.randint(0, 2, n)

    # Run local search on spectral init
    labels, cut = _run_local_search(init_labels, A, deg)
    
    return labels
