import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a valid graph with n=40
    for n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26]:
        if time.time() > deadline:
            break
        if n == 26:
            # Fallback to the seed solution
            return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
        
        # Randomized greedy construction
        best_graph = None
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Initialize empty graph
            adj = np.zeros((n, n), dtype=bool)
            
            # Add edges randomly
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        # Check if adding (i,j) creates a triangle
                        if not np.any(adj[i] & adj[j]):
                            adj[i, j] = True
                            adj[j, i] = True
            
            # Local search to reduce independence number
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n):
                    for j in range(i+1, n):
                        if time.time() > deadline:
                            break
                        # Try flipping edge (i,j)
                        old_val = adj[i, j]
                        if old_val:
                            # Remove edge
                            adj[i, j] = False
                            adj[j, i] = False
                        else:
                            # Add edge if no triangle is created
                            if not np.any(adj[i] & adj[j]):
                                adj[i, j] = True
                                adj[j, i] = True
                            else:
                                continue
                        
                        # Check independence number
                        # Use a heuristic: maximum independent set size
                        # If the maximum independent set is < 10, it's valid
                        # We can approximate this by checking if there's an independent set of size 10
                        # For speed, we can use a simple greedy approach to find a large independent set
                        # If the greedy finds a set of size 10, it's invalid
                        # If the greedy finds a set of size < 10, it might still be invalid, but it's a good heuristic
                        
                        # To be exact, we can use a branch and bound, but it's slow
                        # For now, use a greedy heuristic
                        independent_set = []
                        remaining = set(range(n))
                        while remaining:
                            # Pick the vertex with the minimum degree in the induced subgraph
                            min_deg = float('inf')
                            min_v = None
                            for v in remaining:
                                deg = np.sum(adj[v, list(remaining)])
                                if deg < min_deg:
                                    min_deg = deg
                                    min_v = v
                            independent_set.append(min_v)
                            remaining.remove(min_v)
                            for u in remaining:
                                if adj[min_v, u]:
                                    remaining.remove(u)
                        
                        if len(independent_set) < 10:
                            improved = True
                            break
                        else:
                            # Revert the change
                            adj[i, j] = old_val
                            adj[j, i] = old_val
                    
                    if time.time() > deadline:
                        break
                
                if time.time() > deadline:
                    break
            
            # Final check with exact branch and bound for independence number
            if time.time() < deadline:
                if not has_independent_set_10(adj, n):
                    edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j]]
                    return {'n': n, 'edges': edges}
    
    # Fallback to the seed solution
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def has_independent_set_10(adj, n):
    # Branch and bound to check if there's an independent set of size 10
    # This is a simplified version for speed
    # If it's too slow, we can return False and rely on the heuristic
    # For now, use a simple DFS
    def dfs(remaining, current_set):
        if len(current_set) >= 10:
            return True
        if not remaining:
            return False
        
        # Pick a vertex
        v = next(iter(remaining))
        remaining.remove(v)
        
        # Include v
        new_remaining = {u for u in remaining if not adj[v, u]}
        if dfs(new_remaining, current_set | {v}):
            return True
        
        # Exclude v
        if dfs(remaining, current_set):
            return True
        
        remaining.add(v)
        return False
    
    return dfs(set(range(n)), set())
