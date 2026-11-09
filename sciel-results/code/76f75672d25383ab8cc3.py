"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using Multistart Greedy Local Search.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    # Precompute degrees for faster gain calculation
    degrees = adj.sum(axis=1)
    
    def get_cut_size(labels):
        """Calculate cut size."""
        mask_0 = (labels == 0)
        mask_1 = (labels == 1)
        return np.sum(adj[np.ix_(mask_0, mask_1)])

    def local_search(labels, max_passes=100):
        """
        Greedy Local Search: repeatedly flip vertices if it improves the cut.
        """
        labels = labels.copy()
        current_cut = get_cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        for _ in range(max_passes):
            improved = False
            for v in range(n):
                # Calculate gain of flipping v
                # s = edges to same side
                # o = edges to opposite side
                # gain = s - o
                # s + o = degrees[v]
                # gain = 2*s - degrees[v]
                
                same_side_mask = (labels == labels[v])
                s = np.dot(adj[v], same_side_mask)
                gain = 2 * s - degrees[v]
                
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    improved = True
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
            
            if not improved:
                break
        
        return best_labels, best_cut

    # Multistart
    best_labels = None
    best_cut = -1
    
    start_time = time.time()
    restarts = 0
    max_restarts = 5000 # High cap, limited by time
    
    while restarts < max_restarts:
        if time.time() - start_time > 115: # Leave 5s buffer
            break
        
        # Initialization strategies
        r = restarts
        if r % 3 == 0:
            labels = np.random.randint(0, 2, n)
        elif r % 3 == 1:
            # Alternating
            labels = np.arange(n) % 2
        else:
            # Random bias
            labels = (np.random.rand(n) > 0.4).astype(int)
            
        labels, cut = local_search(labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        restarts += 1
        
    if best_labels is None:
        best_labels = np.random.randint(0, 2, n)
        
    return best_labels
# EVOLVE-BLOCK-END
