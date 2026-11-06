"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    # --- Pre-computed known valid coloring for n=17 (if available) ---
    # Since finding one from scratch is hard, we rely on search, but we can
    # also try to construct one via a known algebraic structure if possible.
    # However, the task asks to maximize n. The best known is 17.
    # We will attempt to find a valid coloring for n=17, 16, etc.
    
    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for C6 in each color class
        # A C6 is a cycle of length 6.
        # We can check for a path of length 5 between two adjacent vertices
        # that doesn't use the direct edge, or simply look for any cycle of length 6.
        # Efficient way: for each color, for each edge (u,v), check if there is
        # a path of length 5 from u to v in the remaining graph (excluding edge u-v).
        # Actually, simpler: just check if the graph contains a C6.
        # For small n (<=20), we can do this with DFS.
        
        for c in range(4):
            # Build adjacency list for this color
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            # Check for C6
            # A C6 consists of vertices v0, v1, v2, v3, v4, v5 such that
            # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) are edges.
            # We can fix v0 and v1 to be an edge, then search for a path of length 4
            # from v1 to v0 that avoids v0 until the end and doesn't reuse vertices.
            # Wait, the path is v1-v2-v3-v4-v5-v0. So we need a path of length 4 from v1 to v0
            # in the graph where v0 is removed from the intermediate nodes?
            # Let's just do a DFS for a cycle of length 6.
            
            # Optimization: if the number of edges in this color is less than 6, no C6.
            # Also, if max degree is too low, maybe no C6?
            
            # Simple DFS check for C6
            # For each edge (u,v), check if there is a path of length 4 from u to v
            # that does not use u or v as internal nodes and does not use the edge (u,v).
            # Actually, a C6 is u-v-w-x-y-z-u.
            # So if we fix edge (u,v), we need a path v-w-x-y-z-u of length 4.
            # This means w is neighbor of v (not u), x is neighbor of w (not u,v),
            # y is neighbor of x (not u,v,w), z is neighbor of y (not u,v,w,x) and z is neighbor of u.
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u: continue # process each edge once
                    # We have edge (u,v). Look for path v -> w -> x -> y -> z -> u
                    # w in N(v) \ {u}
                    for w in adj_list[v]:
                        if w == u: continue
                        # x in N(w) \ {u, v}
                        for x in adj_list[w]:
                            if x == u or x == v: continue
                            # y in N(x) \ {u, v, w}
                            for y in adj_list[x]:
                                if y == u or y == v or y == w: continue
                                # z in N(y) \ {u, v, w, x} and z in N(u)
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x: continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def attempt(n, time_budget):
        """
        Backtracking search for 4-coloring of K_n without monochromatic C6.
        """
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]

        def has_path_len_5(c, u, v):
            """Check for simple path of length 5 from u to v in color c."""
            # u - x1 - x2 - x3 - x4 - v
            # Iterative DFS
            stack = [(u, 5, 1 << u)]
            while stack:
                cur, depth, visited = stack.pop()
                if depth == 0:
                    if cur == v:
                        return True
                    continue
                if depth == 1:
                    if (adj[c][cur] >> v) & 1:
                        return True
                    continue
                mask = adj[c][cur] & ~visited
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    stack.append((nxt, depth - 1, visited | lsb))
                    mask -= lsb
            return False

        def is_valid(c, u, v):
            return not has_path_len_5(c, u, v)

        stack = [(0, 0)]  # (edge_index, colors_tried_mask)
        while stack:
            idx, tried_mask = stack[-1]

            if time.time() > local_deadline:
                return None

            if idx == num_edges:
                return colors[:]

            u, v = edges[idx]
            found = False
            for c in range(4):
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    stack[-1] = (idx, tried_mask | cbit)
                    stack.append((idx + 1, 0))
                    found = True
                    break

            if not found:
                stack.pop()
                if idx > 0:
                    prev_idx = idx - 1
                    prev_u, prev_v = edges[prev_idx]
                    prev_c = colors[prev_idx]
                    colors[prev_idx] = -1
                    adj[prev_c][prev_u] &= ~(1 << prev_v)
                    adj[prev_c][prev_v] &= ~(1 << prev_u)

        return None

    # Try to find a valid coloring for n=17, 16, 15...
    # We start with n=17. If we find one, we return it.
    # If not, we try n=16, etc.
    
    # Note: The seed program provides n=9.
    # We want to beat 9.
    
    # Strategy:
    # 1. Try n=17 with a significant time budget.
    # 2. If fails, try n=16.
    # 3. If fails, try n=15.
    # 4. Fallback to seed n=9 if all else fails.
    
    # Since n=17 is the known maximum, it might be hard to find via pure backtracking
    # without good heuristics or symmetry breaking.
    # However, the inspiration program did this and got 1.111 (which is 17/15? No, 17/15 is 1.133.
    # Wait, the score is value/seed. Seed value is 9? No, seed program returns n=9.
    # So if we return n=17, score is 17/9 = 1.88?
    # The inspiration score was 1.1111. 1.1111 * 9 = 10. So it found n=10?
    # Or maybe the seed value is different.
    # "The evaluator measures the raw value (higher is better). The search score is value/seed"
    # Seed program returns n=9. So seed value = 9.
    # If inspiration got 1.1111, then value = 1.1111 * 9 = 10.
    # So inspiration found n=10.
    # My goal is to find n=17.
    
    # Let's try n=17.
    res = attempt(17, 50)
    if res:
        if verify(17, res):
            return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 30:
        res = attempt(16, 25)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 15:
        res = attempt(15, 12)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}

    # Try n=14
    if time.time() < hard_deadline - 8:
        res = attempt(14, 6)
        if res:
            if verify(14, res):
                return {'n': 14, 'colors': res}

    # Try n=13
    if time.time() < hard_deadline - 4:
        res = attempt(13, 3)
        if res:
            if verify(13, res):
                return {'n': 13, 'colors': res}

    # Fallback to n=10 (known to be easy)
    if time.time() < hard_deadline - 2:
        res = attempt(10, 1.5)
        if res:
            if verify(10, res):
                return {'n': 10, 'colors': res}

    # Absolute fallback: Seed solution n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    # If even n=9 fails verification (shouldn't happen), return n=1
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
