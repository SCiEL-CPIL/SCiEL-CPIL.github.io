"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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
    deadline = start_time + 55
    
    def find_ham_cycle(rem_edges):
        """Find a Hamiltonian cycle in the graph defined by rem_edges."""
        if not rem_edges:
            return None
        
        # Build adjacency
        rem_adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Check connectivity
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
        
        # Check minimum degree
        for v in range(n):
            if len(rem_adj[v]) < 2:
                return None
        
        # Backtracking with Warnsdorff's heuristic
        path = [0]
        visited = {0}
        found = False
        
        def backtrack(u, depth):
            nonlocal found
            if found:
                return
            
            if depth == n:
                if 0 in rem_adj[u]:
                    found = True
                return
            
            # Get unvisited neighbors
            candidates = [v for v in rem_adj[u] if v not in visited]
            if not candidates:
                return
            
            # Sort by degree (Warnsdorff)
            candidates.sort(key=lambda x: len(rem_adj[x]))
            
            for v in candidates:
                path.append(v)
                visited.add(v)
                backtrack(v, depth + 1)
                path.pop()
                visited.remove(v)
                if found:
                    return
        
        backtrack(0, 1)
        
        if found:
            return path
        return None
    
    def find_ham_cycle_random(rem_edges, start_vertex=None):
        """Find a Hamiltonian cycle with randomization."""
        if not rem_edges:
            return None
        
        rem_adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Check connectivity
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
        
        # Check minimum degree
        for v in range(n):
            if len(rem_adj[v]) < 2:
                return None
        
        if start_vertex is None:
            start_vertex = random.randint(0, n-1)
        
        path = [start_vertex]
        visited = {start_vertex}
        found = False
        
        def backtrack(u, depth):
            nonlocal found
            if found:
                return
            
            if depth == n:
                if start_vertex in rem_adj[u]:
                    found = True
                return
            
            candidates = [v for v in rem_adj[u] if v not in visited]
            if not candidates:
                return
            
            # Randomize order with some bias toward lower degree
            random.shuffle(candidates)
            candidates.sort(key=lambda x: (len(rem_adj[x]), random.random()))
            
            for v in candidates:
                path.append(v)
                visited.add(v)
                backtrack(v, depth + 1)
                path.pop()
                visited.remove(v)
                if found:
                    return
        
        backtrack(start_vertex, 1)
        
        if found:
            return path
        return None
    
    # Search loop with multiple strategies
    while time.time() < deadline:
        rem_edges = set(edge_set)
        cycles = []
        
        # Try to find cycles using different starting vertices and randomizations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Alternate between deterministic and random strategies
            if attempt % 3 == 0:
                c = find_ham_cycle(rem_edges)
            else:
                start_v = random.randint(0, n-1)
                c = find_ham_cycle_random(rem_edges, start_v)
            
            if c is None:
                break
            
            cycles.append(c)
            
            # Remove edges of this cycle
            for i in range(n):
                u = c[i]
                v = c[(i+1)%n]
                e = tuple(sorted((u, v)))
                if e in rem_edges:
                    rem_edges.remove(e)
            
            if len(cycles) > len(best_sol):
                best_sol = cycles.copy()
                if len(best_sol) >= 3:
                    return best_sol
        
        # If we couldn't find 3, try a fresh restart with different random seed
        random.seed()
    
    # Ensure best_sol is not empty
    if not best_sol and valid_seed:
        best_sol = [seed_cycle]
    
    return best_sol
# EVOLVE-BLOCK-END
