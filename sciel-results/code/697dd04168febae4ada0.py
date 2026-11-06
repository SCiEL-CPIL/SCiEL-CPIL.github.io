"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Partitioning and Fast Local Search.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph.
        
    Returns:
        np.array of 0/1 labels per vertex.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    def fast_local_search(labels):
        """
        Fast local search that maintains adj_labels incrementally.
        """
        labels = labels.copy()
        adj_labels = adj @ labels
        
        improved = True
        max_iters = 1000
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Calculate gains for all vertices
            # delta = (1 - 2 * labels[v]) * (degrees[v] - 2 * adj_labels[v])
            gains = (1 - 2 * labels) * (degrees - 2 * adj_labels)
            
            # Find the vertex with the best positive gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain > 0:
                v = max_gain_idx
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update adj_labels for neighbors of v
                # If old_label was 0, new is 1. Neighbors see +1.
                # If old_label was 1, new is 0. Neighbors see -1.
                delta_label = 1 - 2 * old_label
                neighbors = np.where(adj[v] == 1)[0]
                adj_labels[neighbors] += delta_label
                
                improved = True
        
        # Calculate final cut size
        adj_l = adj @ labels
        crossing = labels * degrees + adj_l - 2 * labels * adj_l
        cut = int(np.sum(crossing) / 2)
        
        return labels, cut

    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    L = np.diag(degrees) - adj
    labels_best = None
    cut_best = -1
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Second smallest eigenvector (index 1)
        fiedler_vec = eigvecs[:, 1]
        median_val = np.median(fiedler_vec)
        labels_spectral = (fiedler_vec > median_val).astype(int)
        
        labels_curr, cut_curr = fast_local_search(labels_spectral)
        if cut_curr > cut_best:
            cut_best = cut_curr
            labels_best = labels_curr
    except Exception:
        pass

    # 2. Multi-start Random Initialization
    num_starts = 50
    for i in range(num_starts):
        labels_random = np.random.randint(0, 2, n)
        labels_curr, cut_curr = fast_local_search(labels_random)
        if cut_curr > cut_best:
            cut_best = cut_curr
            labels_best = labels_curr

    if labels_best is None:
        labels_best = np.zeros(n, dtype=int)
        
    return labels_best.astype(int)
# EVOLVE-BLOCK-END
