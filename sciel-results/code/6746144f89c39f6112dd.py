"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque
sys.setrecursionlimit(10000)

def construct():
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    n = P['n']
    edges = [tuple(sorted(e)) for e in P['edges']]
    edge_set = set(edges)
    
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify the natural Hamiltonian cycle
    seed_cycle = list(range(n))
    valid_seed = all(tuple(sorted((seed_cycle[i], seed_cycle[(i+1)%n]))) in edge_set for i in range(n))
    best_sol = [seed_cycle] if valid_seed else []
    
    start_time = time.time()
    deadline = start_time + 55
    
    def find_ham_cycle(rem_edges, start_vertex=0, rng=None):
        """Find a Hamiltonian cycle in the subgraph defined by rem_edges."""
        if len(rem_edges) < n:
            return None
            
        # Build adjacency
        rem_adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        # Check connectivity
        if not rem_edges:
            return None
        visited = {start_vertex}
        stack = [start_vertex]
        while stack:
            u = stack.pop()
            for v in rem_adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < n:
            return None
            
        # Check degrees
        for i in range(n):
            if len(rem_adj[i]) < 2:
                return None
                
        # Backtracking search
        path = [start_vertex]
        visited = {start_vertex}
        
        def backtrack(u, depth):
            if depth == n:
                if start_vertex in rem_adj[u]:
                    return True
                return False
                
            candidates = [v for v in rem_adj[u] if v not in visited]
            if not candidates:
                return False
            
            # Heuristic: sort by degree in remaining graph
            if rng:
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: len(rem_adj[x]) + rng.random() * 0.1)
            else:
                candidates.sort(key=lambda x: len(rem_adj[x]))
            
            for v in candidates:
                path.append(v)
                visited.add(v)
                if backtrack(v, depth + 1):
                    return True
                visited.remove(v)
                path.pop()
            return False
            
        if backtrack(start_vertex, 1):
            return list(path)
        return None

    # Search loop
    attempts = 0
    rng = random.Random(42)
    
    while time.time() < deadline:
        attempts += 1
        rem_edges = set(edge_set)
        cycles = []
        
        while True:
            # Try deterministic first
            c = find_ham_cycle(rem_edges, 0, None)
            if c is None and attempts > 1:
                # Try randomized
                c = find_ham_cycle(rem_edges, 0, rng)
            if c is None:
                # Try different start vertices
                found = False
                for sv in range(1, n):
                    c = find_ham_cycle(rem_edges, sv, None)
                    if c is None and attempts > 1:
                        c = find_ham_cycle(rem_edges, sv, rng)
                    if c is not None:
                        found = True
                        break
                if not found:
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
                if len(best_sol) == 3:
                    return best_sol
                    
        # If we didn't find 3 cycles, try again with different strategy
        if len(cycles) < 3 and time.time() < deadline:
            # Try a different approach: pick edges more carefully
            pass

    return best_sol
# EVOLVE-BLOCK-END
