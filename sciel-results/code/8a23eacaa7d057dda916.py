"""Spectral optimization approach for minimizing max triangle degree sum."""
import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # We want a simple graph on 16 vertices with exactly 65 edges.
    # Start with a balanced bipartite graph K_{8,8} which has 64 edges.
    # Add one more edge within one part to create triangles.
    # The goal is to minimize the maximum degree sum over triangles.
    
    # Strategy: Use coordinate descent / local search on the edge set.
    # Start with K_{8,8} + one edge, then try to improve by swapping edges.
    
    # Initial construction: K_{8,8} + edge (0,1)
    part_a = list(range(8))
    part_b = list(range(8, 16))
    
    edges_set = set()
    for u in part_a:
        for v in part_b:
            edges_set.add((min(u,v), max(u,v)))
    edges_set.add((0, 1))
    
    def compute_score(edges):
        """Compute the maximum degree sum over all triangles."""
        adj = np.zeros((n, n), dtype=np.int8)
        deg = np.zeros(n, dtype=np.int32)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            deg[u] += 1
            deg[v] += 1
        
        max_sum = 0
        # Find all triangles
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    # Check common neighbors
                    common = np.where(adj[i] & adj[j])[0]
                    for k in common:
                        if k > j:  # avoid duplicates
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    current_edges = list(edges_set)
    current_score = compute_score(current_edges)
    
    # Local search: try swapping edges to reduce max triangle degree sum
    # An edge swap: remove edge (u,v), add edge (x,y) where (x,y) not in graph
    # This keeps edge count constant.
    
    import random
    random.seed(42)
    
    best_score = current_score
    best_edges = list(current_edges)
    
    # Precompute adjacency for efficiency
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in current_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    deg = np.zeros(n, dtype=np.int32)
    for u, v in current_edges:
        deg[u] += 1
        deg[v] += 1
    
    def get_score_from_adj(adj, deg):
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common = np.where(adj[i] & adj[j])[0]
                    for k in common:
                        if k > j:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    # Coordinate descent: try removing each edge and adding a non-edge
    # to see if it reduces the max triangle degree sum
    
    improved = True
    max_iterations = 500
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try each edge removal
        for idx in range(len(current_edges)):
            u, v = current_edges[idx]
            
            # Remove edge (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            deg[u] -= 1
            deg[v] -= 1
            
            # Try adding each non-edge
            for x in range(n):
                for y in range(x+1, n):
                    if adj[x, y] == 0 and (x != u or y != v):
                        # Check if (x,y) is a valid non-edge (not the same as removed)
                        if (x == u and y == v) or (x == v and y == u):
                            continue
                        
                        adj[x, y] = 1
                        adj[y, x] = 1
                        deg[x] += 1
                        deg[y] += 1
                        
                        new_score = get_score_from_adj(adj, deg)
                        
                        if new_score < best_score:
                            best_score = new_score
                            # Apply the change
                            current_edges[idx] = (x, y)
                            improved = True
                        else:
                            # Revert
                            adj[x, y] = 0
                            adj[y, x] = 0
                            deg[x] -= 1
                            deg[y] -= 1
            
            # Restore edge (u,v)
            adj[u, v] = 1
            adj[v, u] = 1
            deg[u] += 1
            deg[v] += 1
            
            if improved:
                break
        if improved:
            break
    
    # Convert to list of lists
    final_edges = [[u, v] for u, v in current_edges]
    
    return {'n': n, 'edges': final_edges}
