import time

def construct():
    # Grötzsch graph edges (11 vertices: 0-10)
    # Cycle 0-4
    cycle_edges = [(i, (i + 1) % 5) for i in range(5)]
    # Mycielski vertices 5-9
    # v' connected to neighbors of v
    # 5 (0') connected to 1, 4
    # 6 (1') connected to 2, 5? No, 1's neighbors are 0, 2. So 6 connected to 0, 2.
    # Wait, standard Mycielski: v' connected to N(v).
    # 0's neighbors: 1, 4. So 5 connected to 1, 4.
    # 1's neighbors: 0, 2. So 6 connected to 0, 2.
    # 2's neighbors: 1, 3. So 7 connected to 1, 3.
    # 3's neighbors: 2, 4. So 8 connected to 2, 4.
    # 4's neighbors: 3, 0. So 9 connected to 3, 0.
    # Apex 10 connected to 5, 6, 7, 8, 9.
    
    edges = [list(e) for e in cycle_edges]
    edges += [[5, 1], [5, 4]]
    edges += [[6, 0], [6, 2]]
    edges += [[7, 1], [7, 3]]
    edges += [[8, 2], [8, 4]]
    edges += [[9, 3], [9, 0]]
    edges += [[10, 5], [10, 6], [10, 7], [10, 8], [10, 9]]
    
    # Find an independent set of size 5
    # Known independent set in Grötzsch: {0, 2, 5, 8, 9}? 
    # Check 0-2: No.
    # 0-5: No.
    # 0-8: No.
    # 0-9: Edge (9 connected to 0). So {0,2,5,8,9} is NOT independent.
    
    # Let's find one.
    # Try {0, 2, 6, 8, 10}?
    # 6 connected to 0, 2. So 0-6 edge. No.
    
    # Try {0, 2, 7, 9, 10}?
    # 7 connected to 1, 3.
    # 9 connected to 3, 0.
    # 0-9 edge. No.
    
    # Try {0, 2, 5, 7, 10}?
    # 5 connected to 1, 4, 10.
    # 7 connected to 1, 3.
    # 10 connected to 5.
    # 5-10 edge. No.
    
    # It seems hard to guess. I will write a small search for an independent set of size 5.
    vertices = list(range(11))
    adj = {i: set() for i in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def is_independent(set_v):
        for i in range(len(set_v)):
            for j in range(i+1, len(set_v)):
                if set_v[j] in adj[set_v[i]]:
                    return False
        return True

    # Simple backtracking to find independent set of size 5
    def find_independent_set(size):
        if size == 0:
            return []
        # Try adding vertices
        # Heuristic: pick vertex with min degree?
        # Just brute force for small graph
        # There are C(11, 5) = 462 combinations.
        from itertools import combinations
        for combo in combinations(vertices, size):
            if is_independent(set(combo)):
                return list(combo)
        return None

    ind_set = find_independent_set(5)
    if ind_set is None:
        # Fallback: just return the base graph if no independent set found (should not happen)
        return edges
    
    # Add 5 new vertices 11-15
    # Connect each to ind_set
    for i in range(11, 16):
        for v in ind_set:
            edges.append([i, v])
            
    return edges
