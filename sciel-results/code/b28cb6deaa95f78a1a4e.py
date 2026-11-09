"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
sys.setrecursionlimit(10000)

def construct():
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    n = P['n']
    edges = [tuple(sorted(e)) for e in P['edges']]
    edge_set = set(edges)
    
    # Adjacency list for original graph
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Seed solution
    seed_cycle = list(range(n))
    # Validate seed
    valid_seed = all(tuple(sorted((seed_cycle[i], seed_cycle[(i+1)%n]))) in edge_set for i in range(n))
    best_sol = [seed_cycle] if valid_seed else []
    
    start_time = time.time()
    deadline = start_time + 50
    
    def find_cycle(rem_edge_set, start_time, deadline):
        # Build adjacency for remaining edges
        rem_adj = {i: set() for i in range(n)}
        for u, v in rem_edge_set:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        # Check if graph is connected (basic prune)
        if not rem_edge_set:
            return None
        visited = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in rem_adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < n:
            return None

        # Iterative DFS with Warnsdorff heuristic
        path = [0]
        visited = {0}
        
        def backtrack(u):
            if len(path) == n:
                if 0 in rem_adj[u]:
                    return True
                return False
            
            # Get candidates
            candidates = [v for v in rem_adj[u] if v not in visited]
            if not candidates:
                return False
            
            # Sort by degree (Warnsdorff)
            candidates.sort(key=lambda x: len(rem_adj[x]))
            
            for v in candidates:
                path.append(v)
                visited.add(v)
                if backtrack(v):
                    return True
                visited.remove(v)
                path.pop()
            return False
        
        if backtrack(0):
            return path
        return None

    # Search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        rem_edges = set(edge_set)
        cycles = []
        
        while True:
            c = find_cycle(rem_edges, start_time, deadline)
            if c is None:
                break
            cycles.append(c)
            # Remove edges of c
            for i in range(n):
                u = c[i]
                v = c[(i+1)%n]
                e = tuple(sorted((u, v)))
                if e in rem_edges:
                    rem_edges.remove(e)
            
            if len(cycles) > len(best_sol):
                best_sol = [list(cyc) for cyc in cycles]
                if len(best_sol) == 3: # Max possible given 6-regularity
                    return best_sol
                    
    return best_sol
# EVOLVE-BLOCK-END
