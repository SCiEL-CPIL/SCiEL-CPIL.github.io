"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Build adjacency bitmask for each vertex
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = list(range(min_verts))
    best_distinct_count = 0
    
    # Time limit guard
    start_time = time.time()
    timeout = 80.0  # 90s limit, keep 10s margin
    
    # Precompute all combinations for each size
    # For speed, we'll use a hybrid approach:
    # 1. For small subsets (k=8,9), use vectorized numpy operations
    # 2. For larger subsets, use bitmask operations
    
    def count_distinct_degrees_vectorized(subsets_arr):
        """
        subsets_arr: shape (num_subsets, k) array of vertex indices
        Returns: array of distinct degree counts for each subset
        """
        num_subsets = subsets_arr.shape[0]
        k = subsets_arr.shape[1]
        
        # For each subset, compute degrees
        # degrees[i, j] = degree of vertex subsets_arr[i, j] in subset i
        degrees = np.zeros((num_subsets, k), dtype=np.int32)
        
        for j in range(k):
            for i in range(num_subsets):
                # Count edges from vertex subsets_arr[i,j] to other vertices in subset i
                deg = 0
                for jj in range(k):
                    if jj != j:
                        deg += adj[subset_arr[i, j], subsets_arr[i, jj]]
                degrees[i, j] = deg
        
        # Count distinct degrees for each subset
        distinct_counts = np.zeros(num_subsets, dtype=np.int32)
        for i in range(num_subsets):
            distinct_counts[i] = len(set(degrees[i].tolist()))
        
        return distinct_counts
    
    # Search subset sizes from largest to smallest
    for k in range(n, min_verts - 1, -1):
        if time.time() - start_time > timeout:
            break
            
        # For size k, max possible distinct degrees is k
        # If we already found >= k distinct degrees, no subset of size <= k can beat it
        if k <= best_distinct_count:
            break
            
        if k <= 10:
            # Use vectorized approach for smaller subsets
            all_combos = list(itertools.combinations(range(n), k))
            num_combos = len(all_combos)
            
            # Process in batches to manage memory
            batch_size = 10000
            for batch_start in range(0, num_combos, batch_size):
                if time.time() - start_time > timeout:
                    break
                    
                batch_end = min(batch_start + batch_size, num_combos)
                batch = np.array(all_combos[batch_start:batch_end], dtype=np.int32)
                
                # Compute degrees for each vertex in each subset
                # degrees[i, j] = degree of batch[i, j] in the induced subgraph batch[i]
                degrees = np.zeros((batch.shape[0], k), dtype=np.int32)
                
                for j in range(k):
                    # For each position j, compute degree for all subsets in batch
                    for jj in range(k):
                        if jj != j:
                            # adj[batch[:, j], batch[:, jj]] gives adjacency
                            degrees[:, j] += adj[batch[:, j], batch[:, jj]]
                
                # Count distinct degrees for each subset
                for i in range(batch.shape[0]):
                    distinct_count = len(set(degrees[i].tolist()))
                    if distinct_count > best_distinct_count:
                        best_distinct_count = distinct_count
                        best_subset = batch[i].tolist()
                        
                        if distinct_count == k:
                            return best_subset
        else:
            # Use bitmask approach for larger subsets
            for subset in itertools.combinations(range(n), k):
                if time.time() - start_time > timeout:
                    break
                    
                # Calculate degrees in the induced subgraph using bitmasks
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                    
                degrees = []
                for v in subset:
                    deg = (adj_mask[v] & mask).bit_count()
                    degrees.append(deg)
                    
                distinct_degrees = len(set(degrees))
                
                if distinct_degrees > best_distinct_count:
                    best_distinct_count = distinct_degrees
                    best_subset = list(subset)
                    
                    # If all degrees are distinct for this size, this is optimal for this k
                    if distinct_degrees == k:
                        return best_subset
                    
    return best_subset
# EVOLVE-BLOCK-END
