"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices to maximize independent set size.
    Higher score means the vertex is selected earlier by the greedy decoder.
    
    Strategy:
    - We want to pick vertices that are "cheap" to include (low degree) 
    - But also vertices whose removal eliminates many high-degree neighbors
    - Use a combination of:
      1. k-core number (lower is better - sparse regions)
      2. Degree (lower is better)
      3. Mean neighbor degree (higher is better - removing dense neighbors)
      4. Clustering coefficient (lower is better - less local structure)
    
    The score is designed so that higher values are selected first.
    """
    start_time = time.time()
    DEADLINE = 120.0  # Safety margin from 150s limit
    
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute k-core numbers using peeling algorithm
    def compute_k_core(adj_matrix, n):
        current_deg = adj_matrix.sum(axis=1).astype(int).copy()
        k_core = np.zeros(n, dtype=int)
        removed = np.zeros(n, dtype=bool)
        
        if n == 0:
            return k_core
        
        max_k = int(current_deg.max()) if n > 0 else 0
        
        # For each k from 1 to max_k, peel vertices with degree < k
        for k in range(1, max_k + 1):
            # Find vertices with current degree < k that haven't been removed
            # These vertices cannot be in the k-core
            while True:
                # Find non-removed vertices with degree < k
                candidates = np.where((~removed) & (current_deg < k))[0]
                if len(candidates) == 0:
                    break
                
                # Remove these vertices
                for v in candidates:
                    if not removed[v]:
                        removed[v] = True
                        # Decrement degree of neighbors
                        neighbors = np.where(adj_matrix[v] > 0)[0]
                        for nb in neighbors:
                            if not removed[nb]:
                                current_deg[nb] -= 1
        
        # The k-core number is the highest k such that the vertex was not removed
        # during peeling at level k
        # Actually, we need to track which k each vertex survived
        # Let's redo this more carefully
        
        return k_core
    
    # Better k-core computation
    def compute_k_core_proper(adj_matrix, n):
        if n == 0:
            return np.array([], dtype=int)
        
        current_deg = adj_matrix.sum(axis=1).astype(int).copy()
        k_core = np.zeros(n, dtype=int)
        
        # Use a bucket-based approach
        max_k = int(current_deg.max()) if n > 0 else 0
        
        # Bucket vertices by their current degree
        from collections import deque
        buckets = [deque() for _ in range(n + 1)]
        for i in range(n):
            d = int(current_deg[i])
            if d <= n:
                buckets[d].append(i)
        
        k = 0
        while k <= max_k:
            # Process all vertices with degree == k
            while len(buckets[k]) > 0:
                v = buckets[k].popleft()
                if current_deg[v] < k:
                    continue  # Already processed
                k_core[v] = k
                # Remove v: decrement neighbors
                neighbors = np.where(adj_matrix[v] > 0)[0]
                for nb in neighbors:
                    if current_deg[nb] > k:
                        old_deg = int(current_deg[nb])
                        current_deg[nb] -= 1
                        if old_deg <= n:
                            buckets[old_deg].remove(nb) if nb in buckets[old_deg] else None
                        if int(current_deg[nb]) <= n:
                            buckets[int(current_deg[nb])].append(nb)
            k += 1
        
        return k_core
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        # Return a simple fallback
        return -degrees
    
    # Compute k-core
    k_core = compute_k_core_proper(adj, n)
    
    # Compute mean neighbor degree
    neighbor_degree_sums = adj.dot(degrees)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = neighbor_degree_sums / degrees
        mean_neighbor_degrees = np.nan_to_num(mean_neighbor_degrees, nan=0.0)
    
    # Compute clustering coefficient (approximate)
    def compute_clustering(adj_matrix, n):
        cc = np.zeros(n, dtype=float)
        for i in range(n):
            deg_i = int(adj_matrix[i].sum())
            if deg_i < 2:
                cc[i] = 0.0
                continue
            # Number of edges among neighbors
            neighbors = np.where(adj_matrix[i] > 0)[0]
            sub_adj = adj_matrix[np.ix_(neighbors, neighbors)]
            edges_among_neighbors = sub_adj.sum() / 2.0
            possible_edges = deg_i * (deg_i - 1) / 2.0
            if possible_edges > 0:
                cc[i] = edges_among_neighbors / possible_edges
            else:
                cc[i] = 0.0
        return cc
    
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        # Skip clustering, use simpler score
        scores = -degrees + mean_neighbor_degrees
        return scores
    
    cc = compute_clustering(adj, n)
    
    # Normalize each component to [0, 1] range for combining
    def normalize(arr):
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    # We want:
    # - Low k-core (sparse regions) -> high score
    # - Low degree -> high score  
    # - High mean neighbor degree -> high score
    # - Low clustering -> high score
    
    k_core_norm = normalize(k_core.astype(float))
    deg_norm = normalize(degrees)
    mnd_norm = normalize(mean_neighbor_degrees)
    cc_norm = normalize(cc)
    
    # Combine with weights
    # Weighting: k-core and degree are most important
    weights = {
        'k_core': -0.35,   # Lower k-core is better
        'degree': -0.30,   # Lower degree is better
        'mean_nd': 0.25,   # Higher mean neighbor degree is better
        'clustering': -0.10  # Lower clustering is better
    }
    
    scores = (weights['k_core'] * k_core_norm + 
              weights['degree'] * deg_norm + 
              weights['mean_nd'] * mnd_norm + 
              weights['clustering'] * cc_norm)
    
    return scores

# EVOLVE-BLOCK-END
