"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Initialization and Local Search.
    
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
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        adj_labels = adj @ labels
        crossing = labels * degrees + adj_labels - 2 * labels * adj_labels
        return int(np.sum(crossing) / 2)
    
    def local_search(labels, max_iters=1000):
        """
        Greedy local search: flip vertices that improve the cut.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_iters):
            # Calculate gains for all vertices
            adj_labels = adj @ labels
            gains = (1 - 2 * labels) * (degrees - 2 * adj_labels)
            
            # Find the vertex with the highest positive gain
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            current_cut += max_gain
            
        return labels, current_cut

    def spectral_init(noise_level=0.0):
        """
        Generate initial labels based on the Fiedler vector.
        """
        # Build Laplacian L = D - A
        L = np.diag(degrees) - adj.astype(float)
        
        try:
            # Find eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Index 1 is the Fiedler vector (second smallest eigenvalue)
            fiedler = eigenvectors[:, 1]
            
            # Add noise if requested
            if noise_level > 0:
                fiedler = fiedler + noise_level * np.random.randn(n)
            
            # Threshold at median to balance partition sizes
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(int)
            
            return labels
        except Exception:
            # Fallback to random if spectral fails
            return np.random.randint(0, 2, n)

    best_labels = None
    best_cut = -1
    
    # Multi-start with spectral variations
    # 1. Pure Spectral
    labels = spectral_init(noise_level=0.0)
    labels_ls, cut_ls = local_search(labels)
    if cut_ls > best_cut:
        best_cut = cut_ls
        best_labels = labels_ls.copy()
        
    # 2. Spectral with noise (to escape specific eigenvector symmetries)
    for _ in range(15):
        labels = spectral_init(noise_level=0.1)
        labels_ls, cut_ls = local_search(labels)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()
            
    # 3. Random starts (to cover cases where spectral is misleading)
    for _ in range(5):
        labels = np.random.randint(0, 2, n)
        labels_ls, cut_ls = local_search(labels)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
