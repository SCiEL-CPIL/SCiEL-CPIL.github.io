"""Rewritten program with vectorized simulated annealing to minimize max triangle degree sum."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    def get_degrees(adj):
        return adj.sum(axis=1)
    
    def compute_max_triangle_degree_sum(adj, degrees):
        """Compute max degree sum over all triangles using vectorized operations."""
        # For each triangle (i,j,k), compute deg(i)+deg(j)+deg(k)
        # A triangle exists when adj[i,j] & adj[j,k] & adj[k,i]
        # Use matrix multiplication to find triangles efficiently
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common = adj[i] & adj[j]
                    # Vectorized: get degrees of common neighbors
                    mask = common.astype(bool)
                    if mask.any():
                        common_degrees = degrees[mask]
                        s = degrees[i] + degrees[j] + common_degrees
                        if len(s) > 0:
                            local_max = np.max(s)
                            if local_max > max_sum:
                                max_sum = local_max
        return max_sum
    
    # Initialize with K_{8,8} + one edge
    def make_initial_adj():
        adj = np.zeros((n, n), dtype=np.int8)
        # Bipartite between 0-7 and 8-15
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        # Add one edge within part A
        adj[0, 1] = 1
        adj[1, 0] = 1
        return adj
    
    adj = make_initial_adj()
    degrees = get_degrees(adj)
    best_score = compute_max_triangle_degree_sum(adj, degrees)
    best_adj = adj.copy()
    best_degrees = degrees.copy()
    
    # Precompute all possible edge pairs for moves
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    # Precompute indices for existing/non-existing edges
    edge_idx_map = {}
    for idx, (i, j) in enumerate(all_edges):
        edge_idx_map[(i, j)] = idx
    
    # Simulated annealing with vectorized updates
    temp = 5.0
    cooling = 0.997
    min_temp = 0.01
    
    # Track current edge state as a set of indices
    current_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                current_edge_set.add(edge_idx_map[(i, j)])
    
    while temp > min_temp and time.time() < deadline:
        # Choose a move: swap an existing edge with a non-existing edge
        if not current_edge_set:
            break
        
        # Pick a random existing edge to remove
        remove_idx = random.choice(list(current_edge_set))
        remove_edge = all_edges[remove_idx]
        
        # Pick a random non-existing edge to add
        all_idx = set(range(len(all_edges)))
        non_existing = all_idx - current_edge_set
        if not non_existing:
            break
        add_idx = random.choice(list(non_existing))
        add_edge = all_edges[add_idx]
        
        # Apply move
        ri, rj = remove_edge
        ai, aj = add_edge
        adj[ri, rj] = 0
        adj[rj, ri] = 0
        adj[ai, aj] = 1
        adj[aj, ai] = 1
        
        current_edge_set.remove(remove_idx)
        current_edge_set.add(add_idx)
        
        new_degrees = get_degrees(adj)
        new_score = compute_max_triangle_degree_sum(adj, new_degrees)
        
        delta = new_score - best_score
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            degrees = new_degrees
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
                best_degrees = degrees.copy()
        else:
            # Revert
            adj[ri, rj] = 1
            adj[rj, ri] = 1
            adj[ai, aj] = 0
            adj[aj, ai] = 0
            
            current_edge_set.remove(add_idx)
            current_edge_set.add(remove_idx)
            degrees = get_degrees(adj)
        
        temp *= cooling
    
    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
