"""Rewritten program: SAT-based search for 4-colorings of K_n without monochromatic C6."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 3:
                    return False
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u:
                        continue
                    for w in adj_list[v]:
                        if w == u:
                            continue
                        for x in adj_list[w]:
                            if x == u or x == v:
                                continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w:
                                    continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x:
                                        continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def try_dfs(n, time_budget):
        """Backtracking search with symmetry breaking and pruning."""
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        # Count of edges per color for symmetry breaking
        color_counts = [0] * 4
        
        def has_c6(c, u, v):
            """Check if adding edge (u,v) to color c creates a C6."""
            # A C6 would be u-v-w-x-y-z-u where (u,v) is the new edge
            # So we need a path v-w-x-y-z-u of length 4 in color c
            # where w,x,y,z are distinct and not u or v
            # v's neighbors (excluding u since we're checking before adding)
            # Actually we need to check if there's a path of length 4 from v to u
            # in the current graph of color c, using only vertices other than u and v as internal nodes
            
            # BFS/DFS from v, depth 4, target u, avoiding u and v as internal
            # Path: v -> a -> b -> c2 -> d -> u
            # a in N(v), a != u
            # b in N(a), b not in {u, v, a}
            # c2 in N(b), c2 not in {u, v, a, b}
            # d in N(c2), d not in {u, v, a, b, c2}
            # u in N(d)
            
            # Use iterative deepening or just check all combinations
            # For efficiency, use bitmask-based approach
            
            # Get neighbors of v in color c (excluding u for now, since edge not added yet)
            # Actually, the edge (u,v) is being added, so in the C6, u and v are adjacent
            # The rest of the path v-a-b-c2-d-u must exist in the CURRENT graph
            
            v_mask = adj[c][v] & ~(1 << u)  # neighbors of v excluding u
            if v_mask == 0:
                return False
            
            for a_bit in [1 << i for i in range(n) if (v_mask >> i) & 1]:
                a = a_bit.bit_length() - 1
                a_mask = adj[c][a] & ~(1 << u) & ~(1 << v) & ~a_bit
                if a_mask == 0:
                    continue
                for b_bit in [1 << i for i in range(n) if (a_mask >> i) & 1]:
                    b = b_bit.bit_length() - 1
                    b_mask = adj[c][b] & ~(1 << u) & ~(1 << v) & ~a_bit & ~b_bit
                    if b_mask == 0:
                        continue
                    for c2_bit in [1 << i for i in range(n) if (b_mask >> i) & 1]:
                        c2 = c2_bit.bit_length() - 1
                        c2_mask = adj[c][c2] & ~(1 << u) & ~(1 << v) & ~a_bit & ~b_bit & ~c2_bit
                        if c2_mask == 0:
                            continue
                        for d_bit in [1 << i for i in range(n) if (c2_mask >> i) & 1]:
                            d = d_bit.bit_length() - 1
                            if (adj[c][d] >> u) & 1:
                                return True
            return False

        def is_valid_assignment(c, u, v):
            """Check if assigning color c to edge (u,v) is valid."""
            # Symmetry breaking: first edge must be color 0
            if u == 0 and v == 1:
                return c == 0
            # Edge (0,2): can be 0 or 1
            if u == 0 and v == 2:
                return c in (0, 1)
            # Edge (0,3): can be 0, 1, or 2
            if u == 0 and v == 3:
                return c in (0, 1, 2)
            
            # Check C6 constraint
            return not has_c6(c, u, v)

        # Order edges for better pruning: start with edges from vertex 0
        # This helps with symmetry breaking
        edge_order = list(range(num_edges))
        # Move edges from vertex 0 to the front
        edges_from_0 = [i for i in range(num_edges) if edges[i][0] == 0]
        other_edges = [i for i in range(num_edges) if edges[i][0] != 0]
        edge_order = edges_from_0 + other_edges
        
        # Map from original edge index to position in search order
        pos_to_edge = [None] * num_edges
        edge_to_pos = [None] * num_edges
        for pos, eidx in enumerate(edge_order):
            pos_to_edge[pos] = eidx
            edge_to_pos[eidx] = pos
        
        stack = [(0, 0)]  # (position in edge_order, tried colors mask)
        
        while stack:
            if time.time() > local_deadline:
                return None
            
            pos, tried_mask = stack[-1]
            
            if pos == num_edges:
                # Found a complete assignment
                return colors[:]
            
            eidx = pos_to_edge[pos]
            u, v = edges[eidx]
            
            found = False
            for c in range(4):
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                if not is_valid_assignment(c, u, v):
                    continue
                
                # Assign
                colors[eidx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                color_counts[c] += 1
                
                stack[-1] = (pos, tried_mask | cbit)
                stack.append((pos + 1, 0))
                found = True
                break
            
            if not found:
                stack.pop()
                if pos > 0:
                    prev_pos = pos - 1
                    prev_eidx = pos_to_edge[prev_pos]
                    prev_u, prev_v = edges[prev_eidx]
                    prev_c = colors[prev_eidx]
                    if prev_c >= 0:
                        colors[prev_eidx] = -1
                        adj[prev_c][prev_u] &= ~(1 << prev_v)
                        adj[prev_c][prev_v] &= ~(1 << prev_u)
                        color_counts[prev_c] -= 1
        
        return None

    def try_sat_like(n, time_budget):
        """
        A more structured approach: build the coloring layer by layer.
        For each color, decide which edges get that color, ensuring no C6.
        This is essentially the same as backtracking but with a different structure.
        """
        # This is essentially the same as try_dfs, so we'll just use try_dfs
        return try_dfs(n, time_budget)

    # Try to find valid colorings for decreasing n
    # Start with n=17 (the known maximum)
    for target_n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > hard_deadline - 5:
            break
        
        budget = min(110.0 - (time.time() - start_time), 
                     max(1.0, 110.0 / (18 - target_n)))
        
        res = try_dfs(target_n, budget)
        if res is not None:
            if verify(target_n, res):
                return {'n': target_n, 'colors': res}
    
    # Fallback: n=9 seed solution
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(min(j - i, n - j + i) - 1)
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
