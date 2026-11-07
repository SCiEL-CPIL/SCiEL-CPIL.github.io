"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    deadline = time.time() + 50  # Safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Adjacency list for the base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute candidate edges: |u-v| >= 3
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and abs(u - v) >= 3:
                candidates.append((u, v))

    def is_triangle_free_addition(u, v, adj):
        """Check if adding (u,v) to adj keeps graph triangle-free."""
        # A triangle is formed if there is a w such that w is neighbor of u and w is neighbor of v
        # We check intersection of neighbors
        # Optimization: iterate over smaller neighbor set
        nu = adj[u]
        nv = adj[v]
        if len(nu) > len(nv):
            nu, nv = nv, nu
        for w in nu:
            if w in nv:
                return False
        return True

    def compute_diameter(adj):
        """Compute the diameter of the graph using BFS from each vertex."""
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            for d in dist:
                if d > max_dist:
                    max_dist = d
        return max_dist

    # Try to find a solution with k edges, for k = 1, 2, ...
    for k in range(1, 15):
        if time.time() > deadline:
            break
            
        # Greedy approach for fixed k
        # We try a few random orderings or just the standard greedy
        # Standard greedy: pick best edge at each step
        
        # To improve success rate, we can try multiple random restarts of the greedy selection
        # But for small k, standard greedy might suffice.
        
        found_solution = None
        
        # Try standard greedy
        adj_temp = [set(s) for s in adj]
        added_edges_temp = []
        added_set_temp = set()
        
        current_diam = compute_diameter(adj_temp)
        
        for i in range(k):
            best_edge = None
            best_diam = current_diam + 1
            
            for u, v in candidates:
                if (u, v) in added_set_temp:
                    continue
                if not is_triangle_free_addition(u, v, adj_temp):
                    continue
                
                # Try adding
                adj_temp[u].add(v)
                adj_temp[v].add(u)
                
                new_diam = compute_diameter(adj_temp)
                
                # Remove
                adj_temp[u].remove(v)
                adj_temp[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
            
            if best_edge is None:
                break # Can't add any more valid edges
                
            u, v = best_edge
            adj_temp[u].add(v)
            adj_temp[v].add(u)
            added_set_temp.add((u, v))
            added_edges_temp.append([u, v])
            current_diam = best_diam
            
        if current_diam <= 4:
            found_solution = added_edges_temp
            break
            
        # If standard greedy failed, try a few random restarts
        if found_solution is None:
            import random
            for _ in range(20):
                if time.time() > deadline:
                    break
                random.shuffle(candidates)
                adj_temp = [set(s) for s in adj]
                added_edges_temp = []
                added_set_temp = set()
                current_diam = compute_diameter(adj_temp)
                
                success = True
                for i in range(k):
                    best_edge = None
                    best_diam = current_diam + 1
                    
                    for u, v in candidates:
                        if (u, v) in added_set_temp:
                            continue
                        if not is_triangle_free_addition(u, v, adj_temp):
                            continue
                        
                        adj_temp[u].add(v)
                        adj_temp[v].add(u)
                        new_diam = compute_diameter(adj_temp)
                        adj_temp[u].remove(v)
                        adj_temp[v].remove(u)
                        
                        if new_diam < best_diam:
                            best_diam = new_diam
                            best_edge = (u, v)
                    
                    if best_edge is None:
                        success = False
                        break
                        
                    u, v = best_edge
                    adj_temp[u].add(v)
                    adj_temp[v].add(u)
                    added_set_temp.add((u, v))
                    added_edges_temp.append([u, v])
                    current_diam = best_diam
                    
                if success and current_diam <= 4:
                    found_solution = added_edges_temp
                    break
                    
    if found_solution:
        return [[int(u), int(v)] for u, v in found_solution]
    else:
        # Fallback: return empty or best effort if no solution found (should not happen)
        return []
# EVOLVE-BLOCK-END
