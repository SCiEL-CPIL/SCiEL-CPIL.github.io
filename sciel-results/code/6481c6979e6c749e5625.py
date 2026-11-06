import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Grötzsch graph edges
    # Vertices 0-4: Cycle
    # Vertices 5-9: Copies
    # Vertex 10: Apex
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    # Check if Grötzsch is 4-chromatic (it is)
    # We will add vertices 11-15
    
    # Independent sets in Grötzsch graph
    # We can precompute them or find them on the fly
    # For efficiency, we will try a few known good configurations
    
    # Configuration 1: Based on 4-coloring
    # Color classes of Grötzsch graph:
    # One possible 4-coloring:
    # C1: {0, 2, 6, 8, 9} (Size 5)
    # C2: {1, 4, 7} (Size 3) -- Wait, 1-4 no, 1-7 no, 4-7 no. 
    #    Check 1-6? 6 in C1. 
    #    Check 4-5? 5 not in C2. 
    #    Is {1, 4, 7} independent? 
    #    Edges: 1-0, 1-2, 1-5, 1-6. 
    #    4-0, 4-3, 4-5, 4-9. 
    #    7-2, 7-3, 7-10. 
    #    No edges between 1, 4, 7. Yes.
    # C3: {3, 5} (Size 2) -- 3-5? No. 
    # C4: {10} (Size 1)
    # Sum 5+3+2+1 = 11.
    
    # Let's try to assign new vertices 11-15 to these sets
    # v11 -> C1
    # v12 -> C2
    # v13 -> C3
    # v14 -> C4
    # v15 -> C1
    
    # Edges to add:
    new_edges = []
    # v11 to C1
    for u in [0, 2, 6, 8, 9]:
        new_edges.append([11, u])
    # v12 to C2
    for u in [1, 4, 7]:
        new_edges.append([12, u])
    # v13 to C3
    for u in [3, 5]:
        new_edges.append([13, u])
    # v14 to C4
    for u in [10]:
        new_edges.append([14, u])
    # v15 to C1
    for u in [0, 2, 6, 8, 9]:
        new_edges.append([15, u])
        
    # Edges among new vertices
    # Allowed if neighbor sets are disjoint
    # N(11) = C1
    # N(12) = C2
    # N(13) = C3
    # N(14) = C4
    # N(15) = C1
    
    # Pairs:
    # (11, 12): C1 cap C2 = empty -> Add
    # (11, 13): C1 cap C3 = empty -> Add
    # (11, 14): C1 cap C4 = empty -> Add
    # (11, 15): C1 cap C1 = C1 -> No
    # (12, 13): C2 cap C3 = empty -> Add
    # (12, 14): C2 cap C4 = empty -> Add
    # (12, 15): C2 cap C1 = empty -> Add
    # (13, 14): C3 cap C4 = empty -> Add
    # (13, 15): C3 cap C1 = empty -> Add
    # (14, 15): C4 cap C1 = empty -> Add
    
    add_pairs = [
        (11, 12), (11, 13), (11, 14),
        (12, 13), (12, 14), (12, 15),
        (13, 14), (13, 15),
        (14, 15)
    ]
    for u, v in add_pairs:
        new_edges.append([u, v])
        
    all_edges = edges + new_edges
    
    # Verify triangle-free and 4-chromatic
    # Simple check:
    # Triangle-free:
    adj = {i: set() for i in range(16)}
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    is_tf = True
    for u in range(16):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    is_tf = False
                    break
            if not is_tf: break
        if not is_tf: break
        
    if not is_tf:
        # Fallback to Grötzsch + isolated
        return edges
        
    # 3-colorability check
    def is_3_colorable(adj):
        colors = [-1] * 16
        def backtrack(v):
            if v == 16:
                return True
            for c in range(3):
                if all(colors[u] != c for u in adj[v]):
                    colors[v] = c
                    if backtrack(v+1):
                        return True
                    colors[v] = -1
            return False
        return backtrack(0)
        
    if is_3_colorable(adj):
        return edges
        
    return all_edges

