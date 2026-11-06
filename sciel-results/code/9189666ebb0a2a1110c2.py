import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # 1. Initialize with Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i, j] = True
                adj[j, i] = True
    
    # Helper to get edges list
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges
    
    # Helper to check if a set of vertices contains a K4
    # Returns True if K4 exists
    def has_k4(adj, vertices):
        # vertices is a list of indices
        # Check all 4-subsets
        from itertools import combinations
        for combo in combinations(vertices, 4):
            a, b, c, d = combo
            if adj[a, b] and adj[a, c] and adj[a, d] and adj[b, c] and adj[b, d] and adj[c, d]:
                return True
        return False

    # Helper to check if a set of vertices contains an independent set of size 4 (bar-K4)
    def has_bar_k4(adj, vertices):
        from itertools import combinations
        for combo in combinations(vertices, 4):
            a, b, c, d = combo
            if not adj[a, b] and not adj[a, c] and not adj[a, d] and not adj[b, c] and not adj[b, d] and not adj[c, d]:
                return True
        return False

    # We will try to add vertices one by one.
    # For each new vertex, we need to find a subset S of existing vertices to connect to.
    # Constraints:
    # 1. S must not contain a K4 (else new vertex + K4 = K5)
    # 2. V \ S must not contain a bar-K4 (else new vertex + bar-K4 = bar-K5)
    
    # To make this fast, we can precompute K4s and bar-K4s in the current graph.
    # But as we add vertices, these change.
    # Given n is small (<45), we can just check constraints on the fly for candidate S.
    
    # Heuristic for finding S:
    # Start with S = all vertices, or S = empty, and adjust?
    # Or random sampling.
    
    current_n = n
    max_attempts_per_vertex = 1000
    
    while time.time() < deadline:
        new_vertex = current_n
        found = False
        
        # Precompute K4s and bar-K4s in the current graph (0 to current_n-1)
        # This is O(n^4) which is ~40^4 = 2.5M, acceptable for a few iterations?
        # Actually, we only need to check if S contains a K4.
        # We can just iterate through all K4s in the graph and ensure S doesn't contain any.
        
        k4s = []
        bar_k4s = []
        # Generate all K4s
        from itertools import combinations
        for combo in combinations(range(current_n), 4):
            a, b, c, d = combo
            if adj[a, b] and adj[a, c] and adj[a, d] and adj[b, c] and adj[b, d] and adj[c, d]:
                k4s.append(combo)
        # Generate all bar-K4s
        for combo in combinations(range(current_n), 4):
            a, b, c, d = combo
            if not adj[a, b] and not adj[a, c] and not adj[a, d] and not adj[b, c] and not adj[b, d] and not adj[c, d]:
                bar_k4s.append(combo)
        
        # Now find S subset of {0..current_n-1} such that:
        # - No k4 is fully contained in S
        # - No bar_k4 is fully contained in V\S (i.e., S intersects every bar_k4)
        
        # Randomized Greedy:
        # Start with S = set()
        # Add vertices one by one if they don't complete a K4 in S?
        # And ensure we hit all bar_K4s?
        
        # Better: Start with a random S, and fix violations.
        
        for _ in range(max_attempts_per_vertex):
            # Random initialization
            S = set()
            # Bias towards 50% connection?
            for v in range(current_n):
                if random.random() < 0.5:
                    S.add(v)
            
            valid = True
            # Check K4 constraint: S must not contain any K4
            for k4 in k4s:
                if all(v in S for v in k4):
                    valid = False
                    break
            if not valid:
                continue
                
            # Check bar-K4 constraint: V\S must not contain any bar-K4
            # i.e., S must intersect every bar_k4
            for bk4 in bar_k4s:
                if not any(v in S for v in bk4):
                    valid = False
                    break
            
            if valid:
                # Found a valid neighborhood S
                # Add the new vertex
                adj = np.resize(adj, (current_n + 1, current_n + 1))
                # Initialize new row/col to False
                adj[current_n, :] = False
                adj[:, current_n] = False
                for v in S:
                    adj[current_n, v] = True
                    adj[v, current_n] = True
                
                current_n += 1
                found = True
                break
        
        if not found:
            # Try a few more times with different random seeds or break
            # If we fail, we might be stuck.
            # We can try to flip some edges in the existing graph to create space?
            # For now, just break if we fail a few times in a row?
            # Let's just break to avoid wasting time if stuck.
            break
            
    # Return the best graph found
    edges = get_edges(adj, current_n)
    return current_n, edges
