"""Maximum Cut solver using random hyperplane initialization and 2-opt local search."""
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Goemans-Williamson style random hyperplane initialization (random unit vectors projected onto a random hyperplane).
    2. 2-opt local search: repeatedly find the pair of vertices (one in each set) whose swap increases the cut the most.
    3. Multiple restarts with time-bounded execution.
    
    Returns: 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 145.0  # seconds, leave 5s safety margin for return overhead
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert adjacency to numpy array of floats for faster computation
    adj = np.array(adjacency, dtype=np.float32)
    
    # Precompute degree for each vertex
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the cut size: number of edges crossing the partition."""
        l = labels.astype(np.float32)
        return float(l @ adj - l @ (adj @ l))
    
    def two_opt_search(labels, max_passes=200):
        """
        Improve labels by swapping pairs of vertices (one from set 0, one from set 1)
        if the swap increases the cut size.
        
        For a swap of vertex u (in set 0) and vertex v (in set 1):
        - Edges from u to other set-0 vertices: were NOT in cut, become IN cut (+1 each)
        - Edges from u to other set-1 vertices: were IN cut, become NOT in cut (-1 each)
        - Edges from v to other set-0 vertices: were IN cut, become NOT in cut (-1 each)
        - Edges from v to other set-1 vertices: were NOT in cut, become IN cut (+1 each)
        - Edge u-v: was IN cut (different sets), becomes NOT in cut (-1) [if u,v adjacent]
        
        Net gain = (same0_u - diff0_u) + (same1_v - diff1_v) - 2*adj[u,v]
        where same0_u = neighbors of u in set 0 (excluding v, but v is in set 1 so it's fine)
              diff0_u = neighbors of u in set 1
              same1_v = neighbors of v in set 1
              diff1_v = neighbors of v in set 0
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            if time.time() > DEADLINE - 3:
                break
            
            # Compute gain for all possible swaps (u in set 0, v in set 1)
            # For u in set 0:
            #   gain_from_u = (neighbors of u in set 0) - (neighbors of u in set 1)
            # For v in set 1:
            #   gain_from_v = (neighbors of v in set 1) - (neighbors of v in set 0)
            # Total gain for swap (u,v) = gain_from_u + gain_from_v - 2*adj[u,v]
            
            in_set_0 = (labels == 0).astype(np.float32)
            in_set_1 = (labels == 1).astype(np.float32)
            
            # For each vertex u, compute:
            #   neighbors_in_same_set[u] = adj @ in_set_same_as_u
            #   neighbors_in_other_set[u] = adj @ in_set_other_as_u
            #   gain_component[u] = neighbors_in_same_set[u] - neighbors_in_other_set[u]
            
            # For vertices in set 0: same = set 0, other = set 1
            # For vertices in set 1: same = set 1, other = set 0
            
            same_set_counts = np.where(labels == 0, adj @ in_set_0, adj @ in_set_1)
            other_set_counts = degrees - same_set_counts
            gain_components = same_set_counts - other_set_counts
            
            # Now find the best pair (u in set 0, v in set 1) maximizing:
            # gain_components[u] + gain_components[v] - 2 * adj[u,v]
            
            # Get indices of vertices in each set
            set_0_indices = np.where(labels == 0)[0]
            set_1_indices = np.where(labels == 1)[0]
            
            if len(set_0_indices) == 0 or len(set_1_indices) == 0:
                break
            
            # For efficiency, we can't check all pairs if n is large, but for n=100 it's 100*100=10000 max
            # Let's compute the matrix of gains for all pairs
            # gain_matrix[i,j] = gain_components[set_0_indices[i]] + gain_components[set_1_indices[j]] - 2*adj[set_0_indices[i], set_1_indices[j]]
            
            gc_0 = gain_components[set_0_indices]
            gc_1 = gain_components[set_1_indices]
            
            # Build the gain matrix
            # gc_0[:, None] + gc_1[None, :] - 2 * adj[np.ix_(set_0_indices, set_1_indices)]
            adj_sub = adj[np.ix_(set_0_indices, set_1_indices)]
            gain_matrix = gc_0[:, None] + gc_1[None, :] - 2.0 * adj_sub
            
            # Find the maximum gain
            max_gain = np.max(gain_matrix)
            
            if max_gain <= 1e-6:
                break  # No improving swap found
            
            # Find the pair with maximum gain
            flat_idx = np.argmax(gain_matrix)
            i, j = np.unravel_index(flat_idx, gain_matrix.shape)
            
            u = set_0_indices[i]
            v = set_1_indices[j]
            
            # Perform the swap
            labels[u] = 1
            labels[v] = 0
            current_cut += max_gain
        
        return labels, current_cut
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    def random_hyperplane_init():
        """
        Goemans-Williamson style: generate random unit vectors for each vertex,
        then project onto a random hyperplane to get 0/1 labels.
        """
        # Generate random vectors in R^n (or some dimension)
        # For simplicity, use R^n itself
        vectors = rng.standard_normal((n, n)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms
        
        # Random hyperplane: random unit vector
        hyperplane = rng.standard_normal(n).astype(np.float32)
        h_norm = np.linalg.norm(hyperplane)
        if h_norm > 0:
            hyperplane /= h_norm
        
        # Project: label = 1 if dot(v_i, hyperplane) >= 0 else 0
        projections = vectors @ hyperplane
        labels = (projections >= 0).astype(int)
        
        return labels
    
    # Try several initialization strategies
    num_restarts = 0
    max_restarts = 100
    
    while time.time() < DEADLINE - 5 and num_restarts < max_restarts:
        # Choose initialization
        if num_restarts == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif num_restarts == 1:
            # Random
            labels = rng.integers(0, 2, size=n)
        elif num_restarts % 3 == 2:
            # Random hyperplane (Goemans-Williamson style)
            labels = random_hyperplane_init()
        else:
            # Random
            labels = rng.integers(0, 2, size=n)
        
        # 2-opt local search
        labels, cut = two_opt_search(labels, max_passes=100)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        num_restarts += 1
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
