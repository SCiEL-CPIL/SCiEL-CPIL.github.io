import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    for b in range(5):
        points.append((0, 1, b))
    points.append((0, 0, 1))
    
    # Map points to indices 0..30
    point_index = {p: i for i, p in enumerate(points)}
    
    # 2. Construct adjacency matrix for polarity graph
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            # Dot product mod 5
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Verify edge count
    edge_count = np.sum(adj) // 2
    # Should be 90
    
    # Helper: compute neighbors from adj
    def get_neighbors(adj):
        return [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    
    # Helper: count C4s using common neighbor counts
    def count_c4s(adj, neighbors):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                common_count = 0
                for w in neighbors[u]:
                    if adj[w, v] == 1:
                        common_count += 1
                if common_count >= 2:
                    total += common_count * (common_count - 1) // 2
        return total // 2
    
    neighbors = get_neighbors(adj)
    
    # 3. Find best edge to add
    # We want to minimize the number of new C4s.
    # New C4s = number of edges between N(u) and N(v).
    best_delta = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                # Count new C4s if we add edge (i,j)
                # This is the number of edges between neighbors of i and neighbors of j
                count = 0
                for w in neighbors[i]:
                    for x in neighbors[j]:
                        if adj[w, x] == 1:
                            count += 1
                if count < best_delta:
                    best_delta = count
                    best_edge = (i, j)
                    
    # Add the best edge
    if best_edge:
        i, j = best_edge
        adj[i, j] = 1
        adj[j, i] = 1
        neighbors = get_neighbors(adj)
        
    current_c4s = count_c4s(adj, neighbors)
    
    # Optional: Limited local search to see if we can improve further
    # Try swapping an edge from the base graph with two non-edges?
    # Or just try swapping the added edge with another non-edge (already done in step 3).
    # Let's try a simple 1-swap local search: remove an edge, add a non-edge.
    # This keeps the edge count constant.
    # If we start with a good graph, this might help.
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
    
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        if current_c4s <= 0:
            break
            
        # Try random swaps
        for _ in range(100):
            if time.time() > deadline - 5:
                break
                
            if not edges or not non_edges:
                break

            # Pick a random edge to remove
            e_idx = random.randint(0, len(edges)-1)
            u, v = edges[e_idx]
            
            # Pick a random non-edge to add
            ne_idx = random.randint(0, len(non_edges)-1)
            i, j = non_edges[ne_idx]
            
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[i, j] = 1
            adj[j, i] = 1
            
            neighbors = get_neighbors(adj)
            new_c4s = count_c4s(adj, neighbors)
            
            if new_c4s < current_c4s:
                current_c4s = new_c4s
                edges[e_idx] = (i, j)
                non_edges[ne_idx] = (u, v)
                improved = True
            else:
                # Revert
                adj[i, j] = 0
                adj[j, i] = 0
                adj[u, v] = 1
                adj[v, u] = 1
                neighbors = get_neighbors(adj)
                
    # Return edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                result.append([i, j])
                
    return result
