import numpy as np

def _local_search_incremental(labels, A, deg):
    """
    Runs best-improvement greedy local search with incremental updates.
    This is significantly faster than recomputing A @ labels every step.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Initialize same_counts
    # same_counts[i] = number of neighbors of i with the same label as i
    neighbor_sum_1 = A @ labels
    same_counts = np.where(labels == 0, deg - neighbor_sum_1, neighbor_sum_1)
    
    # Precompute neighbor indices for each node to speed up updates
    # For n=100, this is a list of arrays
    neighbors = [np.where(A[i, :] > 0)[0] for i in range(n)]
    
    while True:
        # Calculate gains
        # Gain of flipping i = 2 * same_counts[i] - deg[i]
        gains = 2 * same_counts - deg
        
        # Find the vertex with the maximum gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        # If the best possible flip does not improve the cut, we are at a local optimum
        if max_gain <= 0:
            break
        
        i = max_gain_idx
        old_label = labels[i]
        new_label = 1 - old_label
        
        # Flip the vertex
        labels[i] = new_label
        
        # Update same_counts for i
        # The neighbors of i are partitioned into 'same' (old) and 'diff' (old).
        # After flip, 'same' becomes 'diff' and 'diff' becomes 'same'.
        same_counts[i] = deg[i] - same_counts[i]
        
        # Update same_counts for neighbors of i
        for j in neighbors[i]:
            if labels[j] == old_label:
                # j was same as i (before flip), now diff
                same_counts[j] -= 1
            else:
                # j was diff from i (before flip), now same
                same_counts[j] += 1

    # Calculate final cut size
    # Cut = Total Edges - Same Edges
    # Total Edges = sum(deg) / 2
    # Same Edges = sum(same_counts) / 2
    cut_size = (np.sum(deg) - np.sum(same_counts)) / 2.0
    
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
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization based on the second eigenvector of the graph Laplacian
    try:
        L = np.diag(deg) - A
        eigvals, eigvecs = np.linalg.eigh(L)
        # The second smallest eigenvector (Fiedler vector)
        fiedler = eigvecs[:, 1]
        
        # Partition based on the sign of the Fiedler vector
        # Since eigenvectors are orthogonal to the all-ones vector, their mean is 0.
        # Splitting at 0 is equivalent to splitting at the median for symmetric distributions.
        init_labels = (fiedler > 0).astype(int)
        
        # Fallback if the split is degenerate
        if init_labels.sum() == 0 or init_labels.sum() == n:
            # Use the next eigenvector or random
            init_labels = (eigvecs[:, 2] > 0).astype(int) if n > 2 else np.arange(n) % 2
            if init_labels.sum() == 0 or init_labels.sum() == n:
                init_labels = np.random.randint(0, 2, n)
    except Exception:
        init_labels = np.random.randint(0, 2, n)
        
    # Run local search on spectral init
    labels, cut = _local_search_incremental(init_labels, A, deg)
    best_labels = labels
    best_cut = cut
    
    # To ensure we don't get stuck in a poor local optimum from the spectral init,
    # we will perform a few "perturbed" local searches.
    # Instead of full random restarts, we flip a small random subset of nodes
    # from the spectral solution and re-run local search.
    # This is cheaper than full random and leverages the spectral structure.
    num_perturbations = 5
    for _ in range(num_perturbations):
        perturbed_labels = best_labels.copy()
        # Flip 10% of the nodes randomly
        num_flips = max(1, int(0.1 * n))
        flip_indices = np.random.choice(n, num_flips, replace=False)
        perturbed_labels[flip_indices] = 1 - perturbed_labels[flip_indices]
        
        labels, cut = _local_search_incremental(perturbed_labels, A, deg)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
            
    return best_labels
