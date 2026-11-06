import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin
    n = 16
    target_edges = 65
    
    def score_graph(adj, deg):
        """Compute the max triangle degree-sum from adjacency matrix and degrees."""
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    continue
                for w in range(v+1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    def edges_to_adj(edges):
        adj = np.zeros((n, n), dtype=bool)
        deg = [0]*n
        for u, v in edges:
            adj[u, v] = adj[v, u] = True
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def local_search(adj, deg, edges):
        """Perform greedy 1-swap local search."""
        current_score = score_graph(adj, deg)
        
        # Precompute edge list for faster iteration? 
        # We iterate over all pairs (u,v) and (x,y)
        
        while time.time() < deadline:
            improved = False
            # Iterate over all possible swaps
            # To make it deterministic and thorough, we can iterate all pairs
            # But we need to break early if we find an improvement to restart the scan
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v]:
                        continue # Only add non-edges
                    
                    for x in range(n):
                        for y in range(x+1, n):
                            if not adj[x, y]:
                                continue # Only remove existing edges
                            
                            # Skip if adding (u,v) is same as removing (x,y)
                            if (u == x and v == y) or (u == y and v == x):
                                continue
                            
                            # Perform swap
                            adj[u, v] = adj[v, u] = True
                            adj[x, y] = adj[y, x] = False
                            deg[u] += 1; deg[v] += 1
                            deg[x] -= 1; deg[y] -= 1
                            
                            new_score = score_graph(adj, deg)
                            
                            # Undo
                            adj[u, v] = adj[v, u] = False
                            adj[x, y] = adj[y, x] = True
                            deg[u] -= 1; deg[v] -= 1
                            deg[x] += 1; deg[y] += 1
                            
                            if new_score < current_score:
                                # Accept
                                adj[u, v] = adj[v, u] = True
                                adj[x, y] = adj[y, x] = False
                                deg[u] += 1; deg[v] += 1
                                deg[x] -= 1; deg[y] -= 1
                                current_score = new_score
                                improved = True
                                # Update edges list
                                # Find and remove (x,y)
                                for i, e in enumerate(edges):
                                    if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                                        edges.pop(i)
                                        break
                                edges.append([u, v])
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return adj, deg, edges, current_score

    best_adj = None
    best_deg = None
    best_edges = None
    best_score = float('inf')
    
    # Initial state: K_{8,8} + one edge
    # This is a good starting point
    init_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj, deg = edges_to_adj(init_edges)
    edges = list(init_edges)
    adj, deg, edges, score = local_search(adj, deg, edges)
    
    if score < best_score:
        best_score = score
        best_adj = adj.copy()
        best_deg = deg.copy()
        best_edges = [e[:] for e in edges]
        
    # If we have time, try random restarts or perturbations
    while time.time() < deadline:
        # Create a perturbed version of the best solution
        # Or start from a random graph
        # Let's try perturbing the best solution by doing a few random swaps
        adj_curr = best_adj.copy()
        deg_curr = best_deg.copy()
        edges_curr = [e[:] for e in best_edges]
        
        # Do 5 random swaps to escape local minimum
        for _ in range(5):
            # Pick random non-edge to add
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u < v and not adj_curr[u, v]:
                    break
            # Pick random edge to remove
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x < y and adj_curr[x, y]:
                    break
            
            # Swap
            adj_curr[u, v] = adj_curr[v, u] = True
            adj_curr[x, y] = adj_curr[y, x] = False
            deg_curr[u] += 1; deg_curr[v] += 1
            deg_curr[x] -= 1; deg_curr[y] -= 1
            
            # Update edges list
            for i, e in enumerate(edges_curr):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges_curr.pop(i)
                    break
            edges_curr.append([u, v])
            
        # Run local search on this perturbed state
        adj_out, deg_out, edges_out, score_out = local_search(adj_curr, deg_curr, edges_curr)
        
        if score_out < best_score:
            best_score = score_out
            best_adj = adj_out.copy()
            best_deg = deg_out.copy()
            best_edges = [e[:] for e in edges_out]
            
    return {'n': 16, 'edges': best_edges}
