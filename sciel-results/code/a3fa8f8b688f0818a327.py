import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Define a known valid coloring for n=17 (Z17 construction)
    # This serves as a fallback and baseline.
    # In Z17, color of (i,j) is determined by difference d = (j-i) % 17.
    # We need a mapping of differences to colors {0,1,2,3} such that no color class contains a C6.
    # A standard Z17 coloring for R_4(C6) uses the following partition of differences:
    # Differences in Z17 are 1..16.
    # One known valid assignment (from literature on R_4(C6)):
    # Color 0: {1, 2, 3, 4, 5, 6, 7, 8} -> Wait, this is too dense.
    # Let's use a verified Z17 coloring.
    # Actually, it's safer to use a small hardcoded valid coloring if I can't verify the Z17 one instantly.
    # However, the prompt says "best known coloring has 17 vertices".
    # I will implement a generic backtracking to find a valid coloring for n=17 if needed,
    # but for speed, I will assume a known structure or just try to search for n=18.
    
    # Strategy:
    # 1. Try to find a valid coloring for n=18.
    # 2. If found, return n=18.
    # 3. If not found, try to find a valid coloring for n=19.
    # 4. If not found, return a valid coloring for n=17 (hardcoded or found quickly).
    
    # To ensure we return *something* valid, we first need a valid n=17.
    # Let's use a simple backtracking to find a valid n=17 first, with a tight time limit.
    # If we can't find n=17 quickly, we are in trouble, but n=9 is the seed.
    # Actually, finding n=17 is non-trivial. 
    # The "inspiration 2" mentions `_construct_z17`. 
    # I will implement a fast checker and a search.
    
    # Let's try to search for n=18 directly. If it fails, we need a fallback.
    # A robust fallback is a valid n=9 (from seed) or n=17 if we can find it.
    # Given the time limit, searching for n=17 from scratch might take too long if not optimized.
    # But we can use the seed's n=9 as a guaranteed fallback.
    # However, to beat 1.0, we need n > 9.
    # Let's try to find a valid n=17 or n=18.
    
    # Efficient C6 check using bitmasks
    # For a given color class (set of edges), we want to check if it contains a C6.
    # A C6 is a sequence v0-v1-v2-v3-v4-v5-v0.
    # This is equivalent to finding a cycle of length 6 in the graph formed by that color.
    
    def has_c6_in_color(color_mask, n):
        # color_mask is a list of bitmasks for each vertex, representing neighbors of that color.
        # We need to check if there is a cycle of length 6.
        # A faster way: check if the graph has a cycle of length 6.
        # This is NP-hard in general, but for n<=20, we can use backtracking or specific checks.
        # Actually, for small n, we can just check all permutations? No, too slow.
        # We can use the fact that C6 is a cycle. 
        # A graph contains a C6 if and only if... it's not a simple property.
        # However, we can prune during construction.
        # When adding edge (u, v) of color c, we check if this creates a C6 in color c.
        # A C6 involving the new edge (u,v) must look like u-v-x-y-z-w-u or similar.
        # Specifically, if we add (u,v), a new C6 must use (u,v).
        # The rest of the cycle is a path of length 4 from u to v in the existing graph of color c?
        # No, the path is u-v-...-u. So we need a path of length 5 from u to v?
        # Cycle: u-v-a-b-c-d-u. Edges: (u,v), (v,a), (a,b), (b,c), (c,d), (d,u).
        # So we need to check if there is a path of length 4 from v to u in the graph *excluding* the edge (u,v) but including other edges of color c?
        # Actually, the other edges must be present in the current coloring of color c.
        # So, when adding (u,v) with color c, we check if there exists a path of length 4 from v to u using only edges of color c that are already present.
        # Path: v-a-b-c-d-u. Length 4 edges.
        # This can be checked by matrix multiplication or BFS/DFS with depth limit.
        # For n=20, depth 4 is small.
        
        # Let G_c be the adjacency matrix of color c.
        # We want to know if (G_c)^4 has a non-zero entry at (v,u) or (u,v)?
        # No, (G_c)^k counts walks of length k.
        # If there is a walk of length 4 from v to u, it might not be a simple path.
        # But if there is a simple path, it's a C6.
        # If there is a walk, it might contain smaller cycles.
        # However, if the graph is simple and we are adding edges one by one, we can just do a DFS from v to u with depth 4.
        
        # To make this fast, we precompute adjacency lists for each color.
        # But in backtracking, the graph changes.
        
        # Alternative: Precompute all C6s in K_n.
        # Number of C6s in K_20 is large but manageable?
        # Number of cycles of length 6 in K_n is n! / (6 * (n-6)!).
        # For n=20: 20! / (6 * 14!) = (20*19*18*17*16*15) / 6 = 20*19*18*17*16*15/6 = 20*19*3*17*16*15 = 46,368,000.
        # This is too many to precompute and check all.
        
        # Better approach: When adding edge (u,v) with color c, check if it completes a C6.
        # A C6 containing (u,v) is determined by the other 4 vertices.
        # Let the cycle be u-v-a-b-c-d-u.
        # The edges (v,a), (a,b), (b,c), (c,d), (d,u) must be of color c.
        # So we need to find a path of length 4 from v to u in the graph of color c.
        
        # We can maintain for each color c, the 4th power of the adjacency matrix?
        # Or just do a DFS.
        
        # Given the small depth (4), a DFS from v to u with depth 4 is fast.
        # We need to avoid using the edge (u,v) in the path, which is naturally handled if we check the path in the *previous* state.
        
        pass # Placeholder

    # Let's implement a backtracking search for n=18.
    # We will represent the coloring as a list of integers, one for each pair (i,j).
    # We will use bitmasks for adjacency of each color to speed up checks.
    
    def search(n, time_limit):
        # colors[i*n + j] is the color of edge (i,j) for i<j?
        # Better: a 2D list or a flat list of length n*(n-1)//2.
        # Let's use a flat list `col` of length m = n*(n-1)//2.
        # Index for (i,j) with i<j: i*(n-1) - i*(i+1)//2 + j - i - 1 ?
        # Simpler: map (i,j) to index.
        
        m = n * (n - 1) // 2
        col = [-1] * m
        
        # Adjacency bitmasks for each color
        # adj[c][i] is a bitmask of vertices j such that (i,j) has color c.
        adj = [[0] * n for _ in range(4)]
        
        # Precompute index map
        idx = {}
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                idx[(i,j)] = k
                idx[(j,i)] = k
                k += 1
        
        # Check if adding edge (u,v) with color c creates a C6
        def creates_c6(u, v, c):
            # We need a path of length 4 from v to u in adj[c]
            # Start from v, depth 0. Target u, depth 4.
            # Path: v -> x1 -> x2 -> x3 -> u
            # Edges: (v,x1), (x1,x2), (x2,x3), (x3,u) all in color c.
            # Note: The edge (u,v) is NOT yet added to adj[c] when we call this.
            # So we are checking if the path exists in the current graph.
            
            # DFS from v to u with depth 4
            # visited set to avoid cycles? A path of length 4 in a simple graph doesn't repeat vertices if it's a simple path.
            # But we just need *a* path.
            # Actually, if there is a walk, it implies a path? Not necessarily, but if there's a walk of length 4, there might be a shorter path.
            # If there is a path of length < 4, does it form a C6? No.
            # We specifically need a path of length 4 to form a C6 with (u,v).
            # If there is a path of length 2, v-x-u, then v-x-u-v is a triangle, not C6.
            # So we must ensure the path is simple and length 4.
            
            # Let's do a DFS with depth limit 4 and track visited.
            stack = [(v, 0, {v})]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 4:
                    if node == u:
                        return True
                    continue
                # Get neighbors of node in color c
                neighbors_mask = adj[c][node]
                # Iterate through set bits
                mask = neighbors_mask
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if nxt not in visited:
                        stack.append((nxt, depth + 1, visited | (1 << nxt)))
                    mask -= lsb
            return False
        
        # Backtracking
        def backtrack(k):
            if time.time() > time_limit:
                return False
            if k == m:
                return True
            
            u, v = None, None
            # Find the pair for index k
            # Inverse of index calculation
            # This is slow. Let's precompute the pairs.
            # Pairs list: [(0,1), (0,2), ..., (0,n-1), (1,2), ...]
            
            if k >= len(pairs):
                return True # Should not happen if m is correct
            
            u, v = pairs[k]
            
            for c in range(4):
                if creates_c6(u, v, c):
                    continue
                
                # Assign color c
                col[k] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(k + 1):
                    return True
                
                # Unassign
                col[k] = -1
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
            
            return False
        
        # Precompute pairs
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((i,j))
        
        # Symmetry breaking: Fix color of (0,1) to 0, (0,2) to 1?
        # This reduces search space by factor of 4 * 4 ...
        # Let's fix (0,1) to 0.
        col[0] = 0
        adj[0][0] |= (1 << 1)
        adj[0][1] |= (1 << 0)
        
        if backtrack(1):
            return col
        return None

    # Try n=19 first? No, n=18 is harder than n=17.
    # If we can't find n=18, we should find n=17.
    # But finding n=17 is also hard.
    # Let's try n=18. If it fails, try n=17.
    
    # However, backtracking for n=18 might be too slow in Python even with bitmasks.
    # The C6 check is the bottleneck.
    
    # Let's try a different approach: Local Search / Simulated Annealing
    # Start with a random coloring, then flip colors to reduce the number of C6s.
    # Objective: Minimize total number of monochromatic C6s.
    # If we reach 0, we have a valid coloring.
    
    def count_c6s(col, n):
        # Count monochromatic C6s
        # For each color c, count C6s in the graph of color c.
        # This is expensive to do from scratch.
        # We can update incrementally.
        
        # For now, let's just return a large number if we don't have an efficient counter.
        # Actually, counting C6s is also hard.
        
        # Let's stick to backtracking but optimize the C6 check.
        # The DFS for path length 4 is O(n^4) in worst case, but usually much less.
        # With n=18, n^4 is ~100,000. If we do this for every edge assignment, it might be slow.
        # Number of assignments is huge.
        
        pass

    # Given the complexity, let's use the "Inspiration 2" strategy which uses `_construct_z17`.
    # I will hardcode a valid Z17 coloring.
    # Z17 coloring:
    # Vertices 0..16.
    # Edge (i,j) color depends on (j-i) % 17.
    # We need a set of differences D0, D1, D2, D3 such that no Di contains a C6 pattern.
    # A C6 in Z17 corresponds to differences d1, d2, d3, d4, d5, d6 summing to 0 mod 17.
    # Actually, a cycle v0..v5. Differences are v1-v0, v2-v1, ..., v0-v5.
    # If all edges have color c, then all differences are in D_c.
    # So we need to ensure that there is no sequence of 6 elements from D_c that sums to 0 mod 17.
    
    # Known partition for R_4(C6) in Z17:
    # D0 = {1, 2, 3, 4, 5, 6, 7, 8} ? No, that's 8 elements. 16/4 = 4.
    # The differences are 1..16.
    # A valid partition is:
    # C0: {1, 2, 4, 8}
    # C1: {3, 6, 12, 9}
    # C2: {5, 10, 13, 14}
    # C3: {7, 15, 11, 16}
    # Let's verify if this works.
    # This is a quadratic residue type partition?
    # 1, 2, 4, 8 are powers of 2.
    # 3, 6, 12, 9 are 3 * powers of 2.
    # 5, 10, 20=3, no. 5, 10, 20-17=3? No.
    # 5, 10, 13, 14.
    # 7, 14, 11, 5? No.
    
    # I will use a search to find a valid Z17 coloring if the hardcoded one is wrong.
    # But to save time, I'll try to return n=17 with a known valid structure.
    
    # Let's try to search for n=17 with the backtracking algorithm above.
    # It should be faster than n=18.
    
    best_n = 9
    best_col = None
    
    # Seed n=9
    n_seed = 9
    best_col = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    # Try to improve to n=17
    # We will run the backtracking search for n=17 with a time limit of 50s.
    # If it finds a solution, use it.
    # If not, try n=18 with remaining time? No, n=18 is harder.
    # If n=17 fails, we return n=9.
    
    # Optimization: Use symmetry breaking.
    # Fix colors of (0,1), (0,2), (0,3) to 0, 1, 2.
    
    def search_n(n, time_limit, fixed_colors=None):
        m = n * (n - 1) // 2
        col = [-1] * m
        adj = [[0] * n for _ in range(4)]
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        idx_map = {}
        for k, (u, v) in enumerate(pairs):
            idx_map[(u,v)] = k
            idx_map[(v,u)] = k
            
        def creates_c6(u, v, c):
            # DFS for path of length 4 from v to u in adj[c]
            stack = [(v, 0, 1 << v)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 4:
                    if node == u:
                        return True
                    continue
                mask = adj[c][node]
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if not (visited & (1 << nxt)):
                        stack.append((nxt, depth + 1, visited | (1 << nxt)))
                    mask -= lsb
            return False
        
        # Apply fixed colors
        if fixed_colors:
            for (u, v), c in fixed_colors.items():
                k = idx_map[(u,v)]
                col[k] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
        
        # Determine start index
        start_k = 0
        if fixed_colors:
            # Find the smallest k not in fixed_colors
            fixed_set = set(idx_map[(u,v)] for (u,v) in fixed_colors.keys())
            for k in range(m):
                if k not in fixed_set:
                    start_k = k
                    break
        
        def backtrack(k):
            if time.time() > time_limit:
                return False
            if k == m:
                return True
            
            u, v = pairs[k]
            
            # Heuristic: Try colors that create fewer C6s?
            # Just try 0..3
            
            for c in range(4):
                if creates_c6(u, v, c):
                    continue
                
                col[k] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(k + 1):
                    return True
                
                col[k] = -1
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
            
            return False
        
        if backtrack(start_k):
            return col
        return None

    # Try n=17
    # Symmetry breaking:
    # (0,1)=0, (0,2)=1, (0,3)=2
    fixed = {(0,1): 0, (0,2): 1, (0,3): 2}
    
    time_left = deadline - time.time()
    if time_left > 10:
        res = search_n(17, time.time() + time_left - 5, fixed)
        if res:
            best_n = 17
            best_col = res
        else:
            # Try n=18? Unlikely to succeed if 17 failed, but maybe 17 failed due to bad heuristic.
            # Let's try n=18 with a smaller time limit.
            time_left = deadline - time.time()
            if time_left > 5:
                res = search_n(18, time.time() + time_left - 2, fixed)
                if res:
                    best_n = 18
                    best_col = res

    if best_n == 9:
        return {'n': best_n, 'colors': best_col}
    else:
        return {'n': best_n, 'colors': best_col}
