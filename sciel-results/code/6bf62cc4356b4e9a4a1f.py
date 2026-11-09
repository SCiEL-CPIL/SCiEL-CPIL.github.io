import random
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead
    
    # Try to find a graph for n=35, then 34, etc.
    for n in range(35, 16, -1):
        # Multiple restarts for each n
        for restart in range(200):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random graph with varying density
            p = 0.4 if n >= 30 else 0.5
            adj = (np.random.rand(n, n) < p).astype(np.int8)
            np.fill_diagonal(adj, 0)
            # Symmetrize
            adj = np.triu(adj, 1)
            adj = adj + adj.T
            
            # Local Search
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check for K4
                k4 = find_k4_fast(adj)
                if k4 is not None:
                    # Pick random edge in K4 to remove
                    edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                                   (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                    u, v = random.choice(edges_in_k4)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    continue
                
                # Check for Ind6
                ind6 = find_ind6_fast(adj)
                if ind6 is not None:
                    # Pick random pair in Ind6 to add edge
                    pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
                    u, v = random.choice(pairs)
                    adj[u, v] = 1
                    adj[v, u] = 1
                    continue
                
                # No violations found
                # Verify exactly
                if find_k4_fast(adj) is None and find_ind6_fast(adj) is None:
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
    # Fallback: Return the best known small graph (Paley 17) if all else fails
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def find_k4_fast(adj):
    """Find a K4 using bitset operations for speed."""
    n = adj.shape[0]
    # Precompute neighbor bitsets as Python ints
    neighbors = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        neighbors.append(mask)
    
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                continue
            # Common neighbors of i and j
            common = neighbors[i] & neighbors[j]
            # Check if any edge exists within common
            temp = common
            while temp:
                lsb = temp & (-temp)
                k = lsb.bit_length() - 1
                temp ^= lsb
                # Check if k has any other neighbor in common
                other = (neighbors[k] & common) & ~(1 << k)
                if other:
                    lsb2 = other & (-other)
                    l = lsb2.bit_length() - 1
                    return [i, j, k, l]
    return None

def find_ind6_fast(adj):
    """Find an independent set of size 6 using bitset operations."""
    n = adj.shape[0]
    # Precompute non-neighbor bitsets
    non_neighbors = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i, j]:
                mask |= (1 << j)
        non_neighbors.append(mask)
    
    def dfs(candidates_mask, current_set, depth):
        if depth == 6:
            return current_set[:]
        
        # Pruning: not enough candidates left
        if candidates_mask.bit_count() + depth < 6:
            return None
        
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            # New candidates: non-neighbors of v that are in candidates_mask, excluding v
            new_mask = candidates_mask & non_neighbors[v] & ~(1 << v)
            
            res = dfs(new_mask, current_set + [v], depth + 1)
            if res is not None:
                return res
        return None
    
    all_mask = (1 << n) - 1
    return dfs(all_mask, [], 0)
