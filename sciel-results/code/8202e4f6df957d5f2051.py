import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def solve(n):
        # n vertices
        # edges in lex order
        # color array
        colors = [-1] * (n * (n - 1) // 2)
        
        # G[c][i] is bitmask of neighbors of i in color c
        G = [[0] * n for _ in range(4)]
        
        # Precompute edge indices
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + j - i - 1
        
        # Check if adding edge (i, j) with color c creates a C6
        # C6 must use (i, j).
        # Path of length 4 between i and some neighbor v2 of j in color c.
        # Path must be in vertices < j (since edges with higher index not set)
        # Actually, if we process in lex order, when we set (i, j),
        # all edges (u, v) with u < v and (u, v) < (i, j) are set.
        # A C6 using (i, j) has 5 other edges.
        # If any of those edges has index > (i, j), it's not set yet.
        # So we only check C6s where all other edges have index < (i, j).
        # This implies the other vertices must be "small enough".
        # Specifically, if the cycle is i-j-v2-v3-v4-v5-i,
        # and j is the largest index in the cycle, then all other edges are < (i, j) is not guaranteed?
        # If j is the largest vertex, then all other pairs (u, v) have max(u, v) < j.
        # So their index is definitely < any edge involving j?
        # Index of (u, v) with u < v < j is < index of (i, j) with i < j?
        # Yes, lex order prioritizes first element.
        # So if j is the max vertex in the C6, the check is valid.
        # What if j is not the max vertex?
        # Then some other vertex k > j is in the C6.
        # Then edge (v, k) might have index > (i, j).
        # So such a C6 is NOT yet complete.
        # So we only need to check if (i, j) completes a C6 where j is the max vertex?
        # No, we need to check if (i, j) completes a C6 where ALL edges are already assigned.
        # This is equivalent to: j is the max vertex in the C6?
        # If j is not the max vertex, say k > j is in the C6.
        # The edge connecting k to its neighbor in the C6 might be (k, m).
        # If k > j, then (k, m) has first element k > j (or m > j).
        # In lex order, (k, m) > (i, j) if k > i?
        # Not always.
        # But we can just check: does there exist a path of length 4 between i and v2
        # in the subgraph induced by vertices {0, ..., j-1}?
        # If such a path exists, then the C6 i-j-v2-...-i uses only vertices < j (except j).
        # So all edges are < (i, j) ?
        # Edges are between vertices < j. So their index is < index of any edge involving j?
        # Yes.
        # So checking path in G restricted to 0..j-1 is sufficient and necessary for "newly formed" C6s.
        
        def is_valid_c6(i, j, c):
            # Check if i is at distance 4 from any neighbor v2 of j in color c (v2 < j)
            # Neighbors of j in color c among 0..j-1
            mask_j = G[c][j] & ((1 << j) - 1)
            
            # For each v2 in mask_j (v2 != i):
            # Check if i is at distance 4 from v2 in G[c] restricted to 0..j-1
            
            # Optimization: 
            # Compute set of nodes at distance 4 from i in G[c] restricted to 0..j-1?
            # Or from v2?
            # Let's compute distance 4 from i.
            # N1 = neighbors of i in G[c] restricted to 0..j-1
            mask_i = G[c][i] & ((1 << j) - 1)
            
            # N2
            mask_2 = 0
            m = mask_i
            while m:
                lsb = m & -m
                node = lsb.bit_length() - 1
                mask_2 |= G[c][node] & ((1 << j) - 1)
                m ^= lsb
            
            # N3
            mask_3 = 0
            m = mask_2
            while m:
                lsb = m & -m
                node = lsb.bit_length() - 1
                mask_3 |= G[c][node] & ((1 << j) - 1)
                m ^= lsb
            
            # N4
            mask_4 = 0
            m = mask_3
            while m:
                lsb = m & -m
                node = lsb.bit_length() - 1
                mask_4 |= G[c][node] & ((1 << j) - 1)
                m ^= lsb
            
            # Check if any v2 in mask_j (v2 != i) is in mask_4
            # v2 must be a neighbor of j in color c.
            # And v2 must be at distance 4 from i.
            # So check if (mask_j & mask_4) has any bit set, excluding i.
            # Note: i is a neighbor of j (we are assigning (i, j) color c).
            # So i is in mask_j.
            # Is i at distance 4 from i? No (distance 0).
            # So we just check if mask_j & mask_4 is non-zero.
            # Wait, if i is in mask_4, that means there is a cycle of length 4?
            # No, distance 4 from i to i implies a cycle of length 4?
            # Path i-v3-v4-v5-i? Length 3?
            # Distance 4 means path of 4 edges.
            # If i is in N4(i), there is a cycle of length 4 (i-a-b-c-i).
            # That's a C4, not C6.
            # The C6 would be i-j-v2-...-i.
            # If v2 = i, the path is i-...-i of length 4.
            # Then the cycle is i-j-i-...-i? No.
            # The cycle is i-j-v2-p1-p2-p3-i.
            # If v2=i, then i-j-i-p1-p2-p3-i.
            # Edge (i, j) and (j, i) are the same edge.
            # This forms a cycle i-j-p3-p2-p1-i?
            # Length 5?
            # Let's trace:
            # Cycle: i - j - v2 - p1 - p2 - p3 - i.
            # Edges: (i,j), (j,v2), (v2,p1), (p1,p2), (p2,p3), (p3,i).
            # If v2 = i:
            # Edges: (i,j), (j,i), (i,p1), (p1,p2), (p2,p3), (p3,i).
            # (i,j) and (j,i) are the same edge.
            # So we have a walk i-j-i-p1-p2-p3-i.
            # This is not a simple cycle.
            # So v2 must be != i.
            
            valid_mask = mask_j & mask_4
            # Remove i if present
            valid_mask &= ~(1 << i)
            
            if valid_mask != 0:
                return False
            return True

        # Backtracking
        # We process edges in lex order
        # idx from 0 to E-1
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(colors):
                return True
            
            i = idx // (n - 1) # This is not correct for lex order
            # Correct lex order index to (i, j):
            # k = 0
            # for i in range(n):
            #   for j in range(i+1, n):
            #     edge idx k
            #     k++
            
            # We can precompute (i, j) for each idx
            # Or compute on fly
            # Let's precompute
            pass

        # Precompute edge endpoints
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(colors):
                return True
            
            i, j = edges[idx]
            
            # Symmetry breaking
            # If i == 0 and j == 1: color must be 0
            # If i == 0 and j == 2: color in {0, 1}
            # If i == 0 and j == 3: color in {0, 1, 2}
            # General rule: c(0, j) >= c(0, j-1)?
            # No, that's for vertex ordering.
            # Let's stick to simple fixes for first few edges.
            
            start_c = 0
            if i == 0 and j == 1:
                start_c = 0
            elif i == 0 and j == 2:
                start_c = 0
            elif i == 0 and j == 3:
                start_c = 0
            # We can also enforce c(0, j) is non-decreasing?
            # If we permute vertices 1..n-1, we can sort their colors to 0.
            # So WLOG c(0, 1) <= c(0, 2) <= ... <= c(0, n-1).
            # This is a strong symmetry breaker.
            
            if i == 0:
                if j > 1:
                    prev_c = colors[get_idx(0, j - 1)]
                    if prev_c != -1:
                        start_c = prev_c
            
            for c in range(start_c, 4):
                # Check validity
                # Tentatively add (i, j) to G[c]
                G[c][i] |= (1 << j)
                G[c][j] |= (1 << i)
                
                if is_valid_c6(i, j, c):
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                G[c][i] &= ~(1 << j)
                G[c][j] &= ~(1 << i)
                colors[idx] = -1
                
            return False

        # Start search
        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    # Try n=18 down to 13
    for n in range(18, 12, -1):
        res = solve(n)
        if res:
            return res
        if time.time() > deadline - 10:
            break
            
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
