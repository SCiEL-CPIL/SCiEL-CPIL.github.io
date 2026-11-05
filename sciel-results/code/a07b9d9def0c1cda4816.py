def construct():
    n = 29
    residues = set()
    for x in range(1, 29):
        residues.add((x * x) % 29)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    # Try to extend to n=30
    # We add vertex 29 (index 29)
    # We need to color edges (i, 29) for i in 0..28
    # We will try to find a valid assignment for these 29 edges.
    
    def is_valid_full(colors_list, n_val):
        # Check if the full graph is K2,2,2-free
        # This is expensive, but we only call it on candidates.
        # For n=30, we can optimize by only checking violations involving the new vertex initially,
        # but we must ensure no new violations are created among old vertices (which are already valid).
        # So we only need to check for K2,2,2 that include vertex 29.
        
        # A K2,2,2 containing vertex 29:
        # 29 must be in one of the 3 parts.
        # Let the parts be P1, P2, P3.
        # If 29 in P1, then P1 = {29, u}.
        # P2 = {v, w}, P3 = {x, y}.
        # Edges required:
        # (29, v), (29, w), (u, v), (u, w)
        # (29, x), (29, y), (u, x), (u, y)
        # (v, x), (v, y), (w, x), (w, y)
        
        # We can iterate over all u, v, w, x, y?
        # u is 1 of 29.
        # v, w, x, y are 4 distinct vertices from 0..28.
        # This is still large: 29 * C(28,4) ~ 29 * 20475 ~ 600k checks.
        # Each check is O(1) if we have a lookup table for edges.
        # 600k checks is fast in Python (< 0.1s).
        
        # Build edge lookup for the current candidate
        # colors_list is the full list for n_val=30.
        # We need a function edge(i, j)
        
        edge_map = {}
        idx = 0
        for i in range(n_val):
            for j in range(i + 1, n_val):
                edge_map[(i, j)] = colors_list[idx]
                idx += 1
        
        def get_edge(i, j):
            if i > j: i, j = j, i
            return edge_map.get((i, j), -1) # -1 if not found, but should be found
            
        # Check violations
        # Iterate u (partner of 29 in P1)
        for u in range(29):
            # Iterate P2 = {v, w} from 0..28, v != u
            # Iterate P3 = {x, y} from 0..28, distinct from u, v, w
            # To avoid duplicates, we can enforce ordering v < w and x < y, and sets disjoint.
            
            # We need to check if there exist v, w, x, y such that:
            # 1. Edges (29, v), (29, w), (u, v), (u, w) are all color C
            # 2. Edges (29, x), (29, y), (u, x), (u, y) are all color C
            # 3. Edges (v, x), (v, y), (w, x), (w, y) are all color C
            
            # Note: The color C can be 0 or 1.
            # We can just check if the set of edges are all 0 or all 1.
            
            # Optimization:
            # For a fixed u, let N_C(u) be neighbors of u in color C.
            # We need v, w in N_C(u) such that (29, v) and (29, w) are C.
            # Let S_C = {v in N_C(u) | edge(29, v) == C}
            # We need to pick v, w from S_C.
            # Then we need x, y in N_C(u) \ {v, w} such that (29, x), (29, y) are C.
            # AND edges (v, x), (v, y), (w, x), (w, y) are C.
            
            # This is still complex to optimize simply.
            # Let's just do the 600k checks.
            
            for v in range(29):
                if v == u: continue
                for w in range(v + 1, 29):
                    if w == u: continue
                    
                    # Check P2 edges
                    c_uv = get_edge(u, v)
                    c_uw = get_edge(u, w)
                    c_29v = get_edge(29, v)
                    c_29w = get_edge(29, w)
                    
                    if c_uv != c_uw or c_29v != c_29w or c_uv != c_29v:
                        continue
                        
                    C = c_uv
                    
                    # Now find x, y
                    # x, y must be in N_C(u) and N_C(29)
                    # and x, y distinct from u, v, w
                    candidates = []
                    for k in range(29):
                        if k == u or k == v or k == w: continue
                        if get_edge(u, k) == C and get_edge(29, k) == C:
                            candidates.append(k)
                            
                    # Need to pick x, y from candidates such that
                    # edges (v, x), (v, y), (w, x), (w, y) are C
                    # Iterate pairs in candidates
                    for i in range(len(candidates)):
                        x = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            y = candidates[j]
                            if (get_edge(v, x) == C and get_edge(v, y) == C and 
                                get_edge(w, x) == C and get_edge(w, y) == C):
                                return False # Violation found
                                
        return True

    # Greedy / Random search for the 29 new edges
    # Initialize with random
    import random
    best_colors = None
    
    # Try a few random assignments
    for attempt in range(1000):
        new_edges = [random.randint(0, 1) for _ in range(29)]
        full_colors = colors + new_edges
        if is_valid_full(full_colors, 30):
            return {'n': 30, 'colors': full_colors}
            
    # If random fails, try greedy?
    # Or just return n=29
    return {'n': 29, 'colors': colors}
