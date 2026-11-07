import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    best_added_edges = None
    best_count = float('inf')
    
    def solve_once(strategy):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        added_edges = set()
        
        def is_triangle_free(u, v):
            if adj[u].intersection(adj[v]):
                return False
            return True

        def get_max_diameter():
            max_d = 0
            for i in range(n):
                dist = [-1] * n
                dist[i] = 0
                q = deque([i])
                while q:
                    curr = q.popleft()
                    for neighbor in adj[curr]:
                        if dist[neighbor] == -1:
                            dist[neighbor] = dist[curr] + 1
                            q.append(neighbor)
                d = max(dist)
                if d > max_d:
                    max_d = d
            return max_d

        # Greedy Construction
        while True:
            if time.time() > deadline:
                return None # Timeout
            
            max_d = get_max_diameter()
            if max_d <= 4:
                break
            
            best_edge = None
            best_max_d = max_d + 1
            best_score = -1 # For tie-breaking
            
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    edge = (u, v)
                    if edge in base_edges or edge in added_edges:
                        continue
                    if not is_triangle_free(u, v):
                        continue
                    candidates.append(edge)
                    
            if not candidates:
                break
                
            # Evaluate candidates
            # To speed up, we might not need to evaluate ALL if we have a good heuristic,
            # but N=24 is small.
            for u, v in candidates:
                # Add
                adj[u].add(v)
                adj[v].add(u)
                d = get_max_diameter()
                # Remove
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Determine score
                # Primary: lower d is better
                # Secondary: strategy specific
                if d < best_max_d:
                    best_max_d = d
                    # Calculate tie-break score
                    if strategy == 0: # Longest
                        score = abs(v - u)
                    elif strategy == 1: # Shortest
                        score = -abs(v - u)
                    elif strategy == 2: # Center
                        score = -abs((u + v) / 2.0 - (n - 1) / 2.0)
                    else: # Random
                        score = random.random()
                        
                    best_edge = (u, v)
                    best_score = score
                elif d == best_max_d:
                    if strategy == 0:
                        score = abs(v - u)
                    elif strategy == 1:
                        score = -abs(v - u)
                    elif strategy == 2:
                        score = -abs((u + v) / 2.0 - (n - 1) / 2.0)
                    else:
                        score = random.random()
                    
                    if score > best_score:
                        best_edge = (u, v)
                        best_score = score
            
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added_edges.add(best_edge)

        # Pruning
        changed = True
        while changed:
            if time.time() > deadline:
                return list(added_edges) # Return current state
            changed = False
            for edge in list(added_edges):
                u, v = edge
                adj[u].remove(v)
                adj[v].remove(u)
                added_edges.remove(edge)
                
                if get_max_diameter() <= 4:
                    changed = True
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                    added_edges.add(edge)
                    
        return list(added_edges)

    # Try multiple strategies
    strategies = [0, 1, 2, 3, 3, 3, 3, 3] # Mix of deterministic and random
    for i in range(50): # 50 restarts
        if time.time() > deadline:
            break
        
        # Use a random seed for random strategies
        random.seed(i)
        strat = strategies[i % len(strategies)]
        res = solve_once(strat)
        
        if res is not None:
            if len(res) < best_count:
                best_count = len(res)
                best_added_edges = res
            # If we found a very good solution (e.g., < 5), we might stop early?
            # But let's run all to be sure.
            
    if best_added_edges is None:
        # Fallback: return empty if something went wrong, but task guarantees solution exists
        # The seed program returns many edges, so it's valid.
        # We should ensure we return a valid list.
        # If no solution found (should not happen), return seed's result?
        # No, seed's result is valid but large.
        # Let's just return an empty list if it fails, which is invalid, but better than crashing?
        # Actually, the greedy should always find *a* solution.
        return []
    
    # Convert to list of lists of ints
    return [[int(u), int(v)] for u, v in best_added_edges]
