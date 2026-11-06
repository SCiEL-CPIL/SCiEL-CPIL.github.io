import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Vectorized check for a specific subset
    # adj is an NxN numpy boolean array
    def is_regular_subset_np(subset, adj):
        k = len(subset)
        if k < 6:
            return False
        # Extract subgraph
        # Using advanced indexing
        sub_adj = adj[np.ix_(subset, subset)]
        # Calculate degrees
        degrees = np.sum(sub_adj, axis=1)
        # Check if all degrees are the same
        return np.all(degrees == degrees[0])

    def is_valid_graph_np(adj, n):
        if n < 6:
            return True
        
        # Check all subsets of size >= 6
        # Optimization: If we find a regular subset, return False immediately.
        # We iterate k from 6 to n.
        # For n=20, this is feasible.
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if is_regular_subset_np(subset, adj):
                    return False
        return True

    # Try to find a valid graph for n=20, then 19, etc.
    # We use random graphs with fixed seeds for reproducibility and speed.
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() - start_time > time_limit:
            break
            
        found = False
        best_edges = None
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() - start_time > time_limit:
                break
            
            random.seed(seed)
            # Generate random graph
            adj_list = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_list[i][j] = True
                        adj_list[j][i] = True
            
            # Convert to numpy for fast checking
            adj_np = np.array(adj_list, dtype=bool)
            
            if is_valid_graph_np(adj_np, n):
                found = True
                # Convert back to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_list[i][j]:
                            edges.append([i, j])
                best_edges = edges
                break
        
        if found:
            return {'n': n, 'edges': best_edges}
            
    # Fallback: n=6, K6 minus an edge
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
