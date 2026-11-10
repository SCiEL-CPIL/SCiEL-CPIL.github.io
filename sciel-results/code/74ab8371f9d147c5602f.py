import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency set for fast lookup
    # We need to handle edge removal, so we'll work with sets of edges or a dictionary of adjacency sets
    
    def find_hamiltonian_cycle(adj, n, start=0, deadline=None):
        """
        Finds a Hamiltonian cycle in the graph defined by adj (list of sets).
        Returns a list of vertices or None.
        """
        if deadline is None:
            deadline = time.time() + 5.0
            
        # Warnsdorff's heuristic: sort neighbors by number of unvisited neighbors
        # We'll do a DFS with backtracking
        
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def dfs(u):
            if time.time() > deadline:
                return False
            if len(path) == n:
                # Check if we can close the cycle
                if start in adj[u]:
                    return True
                return False
            
            # Get neighbors of u, sorted by heuristic
            neighbors = list(adj[u])
            # Heuristic: prefer neighbors with fewer available unvisited neighbors
            # This helps avoid dead ends
            def heuristic(v):
                if visited[v]:
                    return 100 # Avoid visited if possible, though we check visited in loop
                unvisited_count = 0
                for w in adj[v]:
                    if not visited[w]:
                        unvisited_count += 1
                return unvisited_count
            
            # We only visit unvisited neighbors
            candidates = [v for v in neighbors if not visited[v]]
            candidates.sort(key=heuristic)
            
            for v in candidates:
                path.append(v)
                visited[v] = True
                if dfs(v):
                    return True
                path.pop()
                visited[v] = False
                
            return False
            
        if dfs(start):
            return path
        return None

    # Try to find 2 edge-disjoint Hamiltonian cycles
    # Strategy:
    # 1. Find C1.
    # 2. Remove edges of C1.
    # 3. Find C2 in remaining graph.
    # 4. If C2 fails, try a different C1.
    
    # Since the graph is small, we can try a few randomizations if the first attempt fails.
    # But first, let's just do a deterministic attempt with good heuristics.
    
    # Initialize adjacency lists
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # We will try up to 5 different starting configurations or random shuffles if needed
    # But deterministic should work for this specific graph if it has 2 disjoint HC.
    
    best_cycles = []
    
    # Attempt 1: Standard
    adj = build_adj(edges)
    c1 = find_hamiltonian_cycle(adj, n, start=0)
    
    if c1:
        # Remove edges of c1
        edges_set = set()
        for i in range(len(c1)):
            u = c1[i]
            v = c1[(i+1) % len(c1)]
            if u > v: u, v = v, u
            edges_set.add((u, v))
        
        # Build new adjacency without c1 edges
        adj2 = [set() for _ in range(n)]
        for u, v in edges:
            if u > v: u, v = v, u
            if (u, v) not in edges_set:
                adj2[u].add(v)
                adj2[v].add(u)
        
        c2 = find_hamiltonian_cycle(adj2, n, start=0)
        
        if c2:
            # Ensure types are correct
            c1 = [int(v) for v in c1]
            c2 = [int(v) for v in c2]
            return [c1, c2]
            
    # If deterministic failed, try a few random restarts with shuffled neighbor order
    # This is a fallback for robustness, though likely unnecessary for this specific graph
    for _ in range(10):
        # Shuffle edges to potentially change the "preferred" path in heuristic
        # Actually, shuffling adjacency lists is better
        adj = build_adj(edges)
        for i in range(n):
            random.shuffle(list(adj[i])) # Note: adj[i] is a set, so we need to convert to list, shuffle, then back? 
            # Wait, find_hamiltonian_cycle uses adj[u] as a set/list. 
            # My find_hamiltonian_cycle converts to list internally and sorts.
            # The sort key is deterministic based on graph structure. 
            # To get different cycles, we can add a random tie-breaker to the heuristic.
            
            # Let's modify the heuristic slightly by adding random noise?
            # Or just try different start vertices?
            # Start vertex doesn't matter for the cycle itself, just the representation.
            
            # Let's try a different approach: Randomized DFS
            # We can modify the heuristic to have a random component.
            
            # For now, if the first deterministic attempt with Warnsdorff fails, 
            # it might be that C1 chosen by Warnsdorff doesn't leave a HC in the remainder.
            # A different C1 might.
            
            # Let's implement a randomized version of the search for C1.
            # We'll just run the search with a random seed affecting the tie-breaking in the sort.
            
            # To keep it simple and within time, we'll just return the best we found so far (which is empty or 1 cycle if we want to be safe? No, must be valid).
            # If we can't find 2, we should return 1? 
            # The task says "Maximize number of cycles". 
            # If we can only find 1, we return 1. 
            # But the seed returns 1. Score 1.0. 
            # We want 2.
            
            # Let's assume the graph allows 2.
            # If my simple DFS fails, I should try harder.
            
            # Let's try a specific heuristic adjustment: 
            # Instead of just sorting by unvisited count, we can shuffle the candidates with the same unvisited count.
            
            # I will re-run the search with a slight randomization in the candidate selection order for ties.
            
            pass # Placeholder for logic below

    # Refined Search:
    # We'll try to find C1, then C2.
    # If C2 fails, we'll try to find a *different* C1.
    # How to find a different C1? 
    # We can remove edges from the graph to force a different path, or use randomized backtracking.
    
    # Given the 90s limit, we can afford a more exhaustive search if needed.
    # But let's first just try the randomized tie-breaking in the DFS.
    
    def find_hamiltonian_cycle_randomized(adj, n, start=0, deadline=None, seed=None):
        if deadline is None:
            deadline = time.time() + 5.0
        if seed is not None:
            random.seed(seed)
            
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def dfs(u):
            if time.time() > deadline:
                return False
            if len(path) == n:
                if start in adj[u]:
                    return True
                return False
            
            candidates = [v for v in adj[u] if not visited[v]]
            
            # Heuristic
            def heuristic(v):
                unvisited_count = 0
                for w in adj[v]:
                    if not visited[w]:
                        unvisited_count += 1
                return unvisited_count
            
            # Group by heuristic value? Or just sort and shuffle ties?
            # Simple: Sort by heuristic, then shuffle the entire list? No, that defeats the heuristic.
            # Sort by (heuristic, random.random())
            candidates.sort(key=lambda v: (heuristic(v), random.random()))
            
            for v in candidates:
                path.append(v)
                visited[v] = True
                if dfs(v):
                    return True
                path.pop()
                visited[v] = False
                
            return False
            
        if dfs(start):
            return path
        return None

    # Try multiple random seeds for the first cycle to increase chances of finding a C1 that leaves a C2
    for attempt in range(20):
        adj = build_adj(edges)
        # Use a different seed for each attempt
        c1 = find_hamiltonian_cycle_randomized(adj, n, start=0, seed=attempt)
        
        if c1:
            edges_set = set()
            for i in range(len(c1)):
                u = c1[i]
                v = c1[(i+1) % len(c1)]
                if u > v: u, v = v, u
                edges_set.add((u, v))
            
            adj2 = [set() for _ in range(n)]
            for u, v in edges:
                if u > v: u, v = v, u
                if (u, v) not in edges_set:
                    adj2[u].add(v)
                    adj2[v].add(u)
            
            # Also randomize the search for C2
            c2 = find_hamiltonian_cycle_randomized(adj2, n, start=0, seed=attempt+100)
            
            if c2:
                c1 = [int(v) for v in c1]
                c2 = [int(v) for v in c2]
                return [c1, c2]
                
    # If we still haven't found 2, return 1 (the best we have, or just fail?)
    # The task says "Maximize number". If we can't find 2, we should return 1 if we found one.
    # But we should have found one in the first attempt.
    
    # Let's ensure we return at least 1 valid cycle if we can't find 2.
    adj = build_adj(edges)
    c1 = find_hamiltonian_cycle(adj, n, start=0)
    if c1:
        return [[int(v) for v in c1]]
        
    return []

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_hamiltonian_cycle(adj, n, start=0, deadline=None, seed=None):
        if deadline is None:
            deadline = time.time() + 5.0
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()
            
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def dfs(u):
            if time.time() > deadline:
                return False
            if len(path) == n:
                if start in adj[u]:
                    return True
                return False
            
            candidates = [v for v in adj[u] if not visited[v]]
            
            def heuristic(v):
                unvisited_count = 0
                for w in adj[v]:
                    if not visited[w]:
                        unvisited_count += 1
                return unvisited_count
            
            # Sort by heuristic, break ties randomly
            candidates.sort(key=lambda v: (heuristic(v), random.random()))
            
            for v in candidates:
                path.append(v)
                visited[v] = True
                if dfs(v):
                    return True
                path.pop()
                visited[v] = False
                
            return False
            
        if dfs(start):
            return path
        return None

    # Try to find 2 edge-disjoint Hamiltonian cycles
    # We try a few random seeds to increase the probability that the chosen C1
    # leaves a subgraph that contains a Hamiltonian cycle C2.
    
    for attempt in range(50):
        adj = build_adj(edges)
        c1 = find_hamiltonian_cycle(adj, n, start=0, seed=attempt)
        
        if c1:
            # Remove edges of c1
            edges_set = set()
            for i in range(len(c1)):
                u = c1[i]
                v = c1[(i+1) % len(c1)]
                if u > v: 
                    u, v = v, u
                edges_set.add((u, v))
            
            # Build adj for remaining edges
            adj2 = [set() for _ in range(n)]
            for u, v in edges:
                if u > v: 
                    u, v = v, u
                if (u, v) not in edges_set:
                    adj2[u].add(v)
                    adj2[v].add(u)
            
            c2 = find_hamiltonian_cycle(adj2, n, start=0, seed=attempt + 1000)
            
            if c2:
                # Convert to list of ints
                c1_int = [int(v) for v in c1]
                c2_int = [int(v) for v in c2]
                return [c1_int, c2_int]
                
    # Fallback: return 1 cycle if we couldn't find 2
    adj = build_adj(edges)
    c1 = find_hamiltonian_cycle(adj, n, start=0)
    if c1:
        return [[int(v) for v in c1]]
        
    return []
# EVOLVE-BLOCK-END
