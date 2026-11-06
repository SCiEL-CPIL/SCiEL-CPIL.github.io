import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Fallback: n=16 Algebraic Construction
    # Vertices are elements of F_2^4 = {0, ..., 15}
    # Edge (u, v) color is determined by the "type" of the difference d = u xor v.
    # We partition the 15 non-zero vectors of F_2^4 into 5 sets of 3.
    # Let the basis be e0, e1, e2, e3.
    # We define 5 "directions" or "classes" of non-zero vectors.
    # Actually, a simpler known construction for R_5(C4) > 16 uses the following:
    # Identify vertices with F_2^4.
    # For u != v, let d = u xor v.
    # We need to map the 15 non-zero d's to 5 colors such that no color class contains a C4.
    # A C4 in color c means u, v, w, x such that u xor v, v xor w, w xor x, x xor u are all in S_c.
    # This is equivalent to u xor v xor w xor x = 0.
    # If we choose S_c to be a "cap" or a set with no 4-term arithmetic progression (in additive group F_2^4),
    # we avoid C4s?
    # Actually, a set S in F_2^m is C4-free if it contains no {a, b, c, d} such that a+b+c+d=0 with a,b,c,d distinct?
    # Wait, the edges are (u, u+s) for s in S.
    # A C4 exists if there are u, v such that u+s1 = v+s2 = u+s3 = v+s4 ...
    # This requires s1 xor s2 = s3 xor s4.
    # So we need S_c to have no distinct s1, s2, s3, s4 such that s1 xor s2 = s3 xor s4.
    # This is equivalent to S_c having no 4 distinct elements summing to 0?
    # s1 xor s2 = s3 xor s4 => s1 xor s2 xor s3 xor s4 = 0.
    # So we need to partition the 15 non-zero vectors of F_2^4 into 5 sets of 3, each sum-free of 4 elements?
    # A set of 3 elements {a, b, c} always has a+b+c != 0 (unless one is 0, but 0 is not in the set).
    # The only way to have 4 distinct elements summing to 0 is if the set has at least 4 elements.
    # Since each color class has size 3, it is IMPOSSIBLE to have 4 distinct elements in a color class.
    # Thus, ANY partition of the 15 non-zero vectors into 5 sets of size 3 yields a C4-free coloring!
    # Wait, is that true?
    # Let's check.
    # Vertices u, v, w, x.
    # Edges (u,v), (v,w), (w,x), (x,u) all color c.
    # u xor v = s1 in S_c
    # v xor w = s2 in S_c
    # w xor x = s3 in S C
    # x xor u = s4 in S_c
    # Sum: u xor v xor v xor w xor w xor x xor x xor u = 0 = s1 xor s2 xor s3 xor s4.
    # So s1 xor s2 xor s3 xor s4 = 0.
    # If S_c has size 3, can we have 4 distinct elements from S_c summing to 0?
    # No, because there are only 3 elements.
    # Can we have s1, s2, s3, s4 not distinct?
    # If s1 = s2, then u xor v = v xor w => u = w. But vertices in a cycle are distinct.
    # If s1 = s3, then u xor v = w xor x.
    # If the cycle is simple (u!=w, v!=x), then s1, s2, s3, s4 must be distinct?
    # In a simple C4 in a complete graph, the vertices are distinct.
    # u, v, w, x distinct.
    # s1 = u xor v. s2 = v xor w.
    # If s1 = s2, then u xor v = v xor w => u = w. Contradiction.
    # So s1 != s2, s2 != s3, s3 != s4, s4 != s1.
    # What about s1 = s3?
    # u xor v = w xor x.
    # This is possible in a C4. e.g., u=0, v=1, w=3, x=2 in F_2^2.
    # 0 xor 1 = 1. 3 xor 2 = 1.
    # s2 = 1 xor 3 = 2. s4 = 2 xor 0 = 2.
    # So s1=s3=1, s2=s4=2.
    # The set of edge labels is {1, 2}.
    # Sum: 1 xor 1 xor 2 xor 2 = 0.
    # So the condition is: S_c must not contain a subset {a, b} such that a xor b = a xor b?
    # The equation is s1 xor s2 xor s3 xor s4 = 0.
    # If s1=s3 and s2=s4, then we just need s1, s2 in S_c.
    # So if |S_c| >= 2, we can form a C4?
    # Let's check the example: S_c = {1, 2}.
    # Vertices 0, 1, 3, 2.
    # Edges: (0,1) label 1. (1,3) label 2. (3,2) label 1. (2,0) label 2.
    # All in S_c.
    # So ANY set S_c with at least 2 elements allows a C4?
    # Yes! In F_2^m, if s1, s2 in S_c, we can always find u, v, w, x such that the edge labels are s1, s2, s1, s2.
    # u=0, v=s1, w=s1 xor s2, x=s2.
    # u xor v = s1.
    # v xor w = s1 xor (s1 xor s2) = s2.
    # w xor x = (s1 xor s2) xor s2 = s1.
    # x xor u = s2.
    # Vertices: 0, s1, s1 xor s2, s2.
    # Are they distinct?
    # 0 != s1 (since s1 != 0).
    # 0 != s2.
    # s1 != s2 (distinct elements in S_c).
    # s1 != s1 xor s2 (since s2 != 0).
    # s2 != s1 xor s2 (since s1 != 0).
    # s1 xor s2 != 0 (since s1 != s2).
    # So yes, if any color class has size >= 2, it contains a C4.
    # Therefore, the algebraic construction over F_2^4 with 5 colors of size 3 DOES NOT WORK for C4-free.
    # It works for K_4-free (no triangles)? No, triangles are u xor v xor w = 0.
    # It works for C4-free only if |S_c| = 1?
    # If |S_c| = 1, say {s}, then we can only have edges of type s.
    # A C4 requires s1 xor s2 xor s3 xor s4 = 0.
    # If s1=s2=s3=s4=s, then s xor s xor s xor s = 0.
    # So even a set of size 1 allows a C4?
    # u=0, v=s, w=0? No, vertices must be distinct.
    # If s1=s2=s3=s4=s, then u xor v = s, v xor w = s => u=w. Not distinct.
    # So if all 4 edges are the same vector s, it forms a path of length 4, not a cycle of distinct vertices?
    # u, v, w, x.
    # u xor v = s
    # v xor w = s => w = v xor s = (u xor s) xor s = u.
    # So w=u. Not a simple cycle.
    # So a set of size 1 is C4-free.
    # A set of size 2 is NOT C4-free.
    # So we cannot partition 15 vectors into 5 sets of 3.
    
    # However, we can use a different algebraic structure.
    # Or, we can use the known fact that R_5(C4) > 16 is achievable.
    # A known construction for n=16 uses the Paley graph or similar?
    # Actually, let's just use a robust backtracking from a random start, but optimized.
    # The previous program failed likely due to weak constraint propagation or bad edge ordering.
    
    # New Strategy:
    # 1. Use a randomized greedy construction to get a "near" solution or a valid small n.
    # 2. Use highly optimized backtracking with:
    #    - Bitmask adjacency.
    #    - Forward checking: Pre-compute forbidden colors for each edge based on current assignment.
    #    - MRV (Minimum Remaining Values) heuristic for edge selection.
    #    - Value ordering: Least Constraining Value.
    
    # Let's implement a fast solver for n=12, 13, 14, 15, 16.
    
    def solve_fast(n, time_limit):
        """
        Optimized backtracking with MRV and bitmasks.
        """
        # State
        # adj[c][u]: bitmask of neighbors of u in color c
        # colors: list of colors for edges in lex order
        # remaining: set of unassigned edges (indices)
        
        # Precompute edge indices
        edges = []
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_idx[(i, j)] = idx
                idx += 1
        
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(5)]
        
        # To speed up MRV, we can maintain a list of unassigned edges.
        # But dynamic MRV is expensive.
        # Let's stick to a fixed order but with strong forward checking.
        # Or, we can use a "queue" of edges and pick the one with fewest valid colors.
        
        # For simplicity and speed in Python, let's use a recursive search with a fixed order
        # but very fast validity checks using bitsets.
        
        # We will use a stack-based approach to avoid recursion limit issues, 
        # but recursion is fine for n<=16 (edges <= 120).
        
        def is_valid(i, j, c):
            # Check if assigning color c to (i, j) creates a C4.
            # 1. Common neighbors of i and j in c must be < 2.
            #    Actually, < 2 is the condition for "no C4 involving i,j as opposite vertices".
            #    Wait, the condition "no two vertices with two common neighbors" is the global one.
            #    Adding (i,j) affects:
            #    - Pair (i,j): common neighbors increase? No, common neighbors are k s.t. (i,k)=c and (j,k)=c.
            #      Adding (i,j) doesn't change the set of common neighbors of i and j (k is distinct from i,j).
            #      So we just need |N_c(i) & N_c(j)| <= 1.
            #    - Pair (i, k) for k in N_c(j):
            #      j becomes a common neighbor of i and k.
            #      So we need |N_c(i) & N_c(k)| == 0 before adding.
            #    - Pair (j, k) for k in N_c(i):
            #      i becomes a common neighbor of j and k.
            #      So we need |N_c(j) & N_c(k)| == 0 before adding.
            
            # Fast check:
            # 1. Check (i, j)
            if (adj[c][i] & adj[c][j]).bit_count() > 1:
                return False
            
            # 2. Check (i, k) for k in N_c(j)
            mask_j = adj[c][j]
            # Iterate over bits in mask_j
            m = mask_j
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                # Check if i and k have any common neighbors in c
                if (adj[c][i] & adj[c][k]) != 0:
                    return False
                m ^= lsb
            
            # 3. Check (j, k) for k in N_c(i)
            mask_i = adj[c][i]
            m = mask_i
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                if (adj[c][j] & adj[c][k]) != 0:
                    return False
                m ^= lsb
            
            return True
        
        # MRV Heuristic:
        # Instead of fixed order, pick the next edge with the fewest valid colors.
        # This requires scanning all unassigned edges.
        # For n=16, 120 edges. Scanning 120 edges * 5 colors * fast check.
        # This is ~600 checks per node. If depth is 120, and branching is low, it's fast.
        
        unassigned = list(range(num_edges))
        
        def backtrack(depth):
            if time.time() > time_limit:
                return False
            
            if not unassigned:
                return True
            
            # Find edge with MRV
            best_idx = -1
            best_valid_colors = []
            min_count = 5
            
            for idx in unassigned:
                i, j = edges[idx]
                valid_cs = []
                for c in range(5):
                    if is_valid(i, j, c):
                        valid_cs.append(c)
                if len(valid_cs) < min_count:
                    min_count = len(valid_cs)
                    best_idx = idx
                    best_valid_colors = valid_cs
                    if min_count == 0:
                        return False # Dead end
                    if min_count == 1:
                        break # Can't get better than 1
            
            # Remove best_idx from unassigned
            unassigned.remove(best_idx)
            
            i, j = edges[best_idx]
            
            # Try valid colors
            # Shuffle to break symmetry
            random.shuffle(best_valid_colors)
            
            for c in best_valid_colors:
                # Assign
                colors[best_idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                if backtrack(depth + 1):
                    return True
                
                # Unassign
                colors[best_idx] = -1
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
            
            # Restore
            unassigned.append(best_idx)
            
            return False
        
        # Run with multiple restarts
        for restart in range(50):
            if time.time() > time_limit:
                break
            random.seed(restart)
            # Reset state
            colors = [-1] * num_edges
            adj = [[0] * n for _ in range(5)]
            unassigned = list(range(num_edges))
            
            if backtrack(0):
                return colors
        
        return None

    # Fallback: n=11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    def check_validity(colors, n):
        if not colors:
            return n <= 2
        if len(colors) != n * (n - 1) // 2:
            return False
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c >= 5:
                    return False
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    common = mask_u & mask_v
                    if common.bit_count() > 1:
                        return False
        return True

    # Try to solve for n = 12 to 16
    # We allocate time proportional to n
    for n in range(12, 17):
        if time.time() > deadline - 5.0:
            break
        
        # Allocate remaining time
        time_left = deadline - time.time()
        # Give more time to smaller n? Or equal time?
        # Let's give 10s to n=12, 15s to n=13, etc.
        # Or just try until deadline.
        
        res = solve_fast(n, deadline - 2.0)
        
        if res is not None:
            if check_validity(res, n):
                best_n = n
                best_colors = res
                # If we found n=16, we can stop or try 17?
                # 17 is likely very hard. Let's stop at 16 if found to ensure we return quickly.
                # But if we have time, we can try 17.
                if n >= 16:
                    break
            else:
                # Should not happen
                pass
        else:
            # If n=12 fails, n=13 is likely to fail or take too long.
            # But maybe n=12 is hard?
            # Let's break if we fail n=12 to save time.
            if n == 12:
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
