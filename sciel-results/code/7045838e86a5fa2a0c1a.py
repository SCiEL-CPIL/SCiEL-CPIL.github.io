import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s limit
    
    # Fallback: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback
    
    # Target n values to try. We want to maximize n.
    # Lazebnik/Woldar gives 26. 
    # If we can't reach 26, we try 25, 24, etc.
    # But greedy backtracking might struggle with very high n if not careful.
    # Let's try from high to low, or low to high?
    # High to low is better for "Maximize n". If we find 26, we stop.
    # If we fail at 26 quickly, we try 25.
    
    # However, backtracking can be exponential. 
    # We need a robust greedy heuristic that rarely backtracks deep.
    
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid coloring for this n
        success = False
        num_attempts = 50 # Multiple random restarts with different vertex orderings/seedings
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Shuffle vertex order to break symmetry? 
            # Or just randomize the greedy choices.
            # Greedy algorithm:
            # Process vertices 0 to n-1.
            # When adding vertex v, assign colors to edges (u, v) for u < v.
            # Constraint: For each color c, the set of neighbors of v in color c (N_c(v)) 
            # must not share more than 1 common neighbor with any u < v?
            # No, the constraint is: For any pair (u, v), they cannot have 2 common neighbors in color c.
            # When adding v, we choose colors for (u, v).
            # For a fixed color c, let S_c be the set of u < v such that (u, v) is color c.
            # For any u in S_c, the common neighbors of u and v in color c are N_c(u) \cap N_c(v).
            # N_c(v) is currently being built (it's S_c).
            # N_c(u) is already fixed.
            # So for each u in S_c, |N_c(u) \cap S_c| must be <= 1.
            # This is a constraint on the set S_c.
            
            # We can model this as: We are assigning colors to edges incident to v.
            # Let's use a backtracking search for the assignment of colors to edges (0,v), (1,v), ..., (v-1,v).
            # There are v edges. 5^v possibilities. For v=25, 5^25 is too big for brute force.
            # But constraints prune heavily.
            
            # State:
            # adj_mask[c][u] : bitmask of neighbors of u in color c.
            # We build vertex by vertex.
            
            adj_mask = [[0] * n for _ in range(5)]
            colors = [0] * (n * (n - 1) // 2)
            edge_map = {}
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    edge_map[(i, j)] = idx
                    idx += 1
            
            # Precompute inverse for speed if needed, but we build in order so we know indices.
            # Actually, we just need to fill `colors` array.
            
            # Greedy/Backtrack for vertex v
            # We process v from 0 to n-1.
            # For v=0, no edges.
            # For v=1, edge (0,1). Color 0-4. No C4 possible with 2 vertices.
            # For v=2, edges (0,2), (1,2).
            
            # Let's implement a recursive backtracking for each new vertex.
            # To make it fast, we use a heuristic: 
            # Try to assign colors such that the "load" is balanced and constraints are satisfied.
            
            # We will use a simple randomized greedy with limited backtracking (depth 2-3) if stuck.
            # Actually, for n=26, a pure greedy might fail. 
            # But the Lazebnik-Woldar construction is algebraic. 
            # Can we generate it directly?
            # The construction is over GF(4) or similar? 
            # Lazebnik-Woldar (2000) "A new lower bound for R(C4, C4) for 5 colors".
            # It uses a projective plane or similar structure.
            # If we can't derive it, we rely on search.
            
            # Let's try a fast local search on the full graph for the current n, 
            # initialized randomly, but with a very fast delta check.
            # This is effectively what the previous program did, but maybe optimized.
            # The prompt asks for a DIFFERENT strategy. 
            # "Greedy constructive algorithm with backtracking".
            
            # Let's implement the vertex-by-vertex backtracking properly.
            
            # Optimization: 
            # When adding vertex v, we assign colors to edges (u, v) for u=0..v-1.
            # We can do this recursively.
            # To speed up, we order u such that those with high degree in remaining colors are constrained first?
            # Or just 0..v-1.
            
            # Let's use a bitmask for "forbidden" checks.
            # When we tentatively assign color c to (u, v):
            # For any other w < v that already has color c to (w, v)? No, we are assigning sequentially.
            # Let's say we are at step k in assigning colors for vertex v. 
            # Edges (0,v)...(k-1,v) are assigned. (k,v) is current.
            # If we assign color c to (k,v):
            # 1. For any u < k such that (u,v) is also color c:
            #    Check if N_c(u) & N_c(v) has more than 1 bit set.
            #    N_c(v) currently contains {0..k-1} assigned to c.
            #    So we check if N_c(u) has any bits in {0..k-1} assigned to c, other than maybe u? No, u is not in N_c(v) unless (u,v) is c.
            #    Wait, N_c(v) is the set of neighbors of v in color c.
            #    If (u,v) is c, then u is in N_c(v).
            #    Common neighbors of u and v in color c are N_c(u) & N_c(v).
            #    We need |N_c(u) & N_c(v)| <= 1.
            #    Since u is in N_c(v), u is in the intersection? No, u is not a neighbor of itself.
            #    So we check if N_c(u) has any common neighbor w with v.
            #    w must be in N_c(v) AND N_c(u).
            #    If there are 2 such w, it's a C4.
            
            # This check is O(1) with bitmasks.
            
            def solve_vertex(v, partial_colors, adj_mask):
                # partial_colors is a list of length v, colors for (0,v)...(v-1,v)
                # adj_mask is the global state for vertices < v
                if v == 0:
                    return True
                if v == 1:
                    # Only one edge (0,1). Any color works.
                    # Randomize for diversity?
                    c = random.randint(0, 4)
                    partial_colors[0] = c
                    adj_mask[c][0] |= (1 << 1)
                    adj_mask[c][1] |= (1 << 0)
                    return True
                
                # Backtracking for vertex v
                # We assign colors to edges (0,v), (1,v), ..., (v-1,v)
                # To improve performance, we can shuffle the order of u?
                # Let's stick to 0..v-1 but use random restarts.
                
                def backtrack(k, current_adj_mask_v):
                    # k is the index of the next vertex u to assign color to (u,v)
                    # current_adj_mask_v[c] is the bitmask of neighbors of v in color c assigned so far
                    if k == v:
                        # All edges assigned
                        # Update global adj_mask
                        for c in range(5):
                            mask = current_adj_mask_v[c]
                            # mask has bits for u < v
                            for u in range(v):
                                if mask & (1 << u):
                                    adj_mask[c][u] |= (1 << v)
                                    adj_mask[c][v] |= (1 << u)
                        # Store colors
                        idx_base = v * (v - 1) // 2 # Index of (0,v) in lexicographic order?
                        # Lex order: (0,1)..(0,n-1), (1,2)..
                        # Index of (u,v) where u < v:
                        # idx = u*n - u*(u+1)//2 + v - u - 1
                        for u in range(v):
                            c = partial_colors[u]
                            idx = u * n - u * (u + 1) // 2 + v - u - 1
                            colors[idx] = c
                        return True
                    
                    # Heuristic: Try colors in random order? Or 0-4?
                    # Random order helps escape local minima in the greedy choice
                    color_order = list(range(5))
                    random.shuffle(color_order)
                    
                    for c in color_order:
                        # Tentatively assign color c to (k, v)
                        # Check constraints
                        valid = True
                        # For any u < k such that (u,v) is color c:
                        # u is in current_adj_mask_v[c]
                        mask_u_v = current_adj_mask_v[c]
                        # We need to check for each u in mask_u_v:
                        # |N_c(u) & N_c(v)| <= 1
                        # N_c(v) is current_adj_mask_v[c] (plus k if we add it)
                        # Actually, N_c(v) after adding k is current_adj_mask_v[c] | (1 << k)
                        new_mask_v = current_adj_mask_v[c] | (1 << k)
                        
                        # Iterate through set bits in mask_u_v
                        m = mask_u_v
                        while m:
                            lsb = m & -m
                            u = lsb.bit_length() - 1
                            m ^= lsb
                            
                            # Check common neighbors of u and v in color c
                            common = adj_mask[c][u] & new_mask_v
                            # u is not in adj_mask[c][u]
                            # If common has > 1 bit, invalid
                            if common.bit_count() > 1:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Also need to check if adding k to N_c(v) creates a C4 with some w < k?
                        # The check above covers u < k.
                        # What about pairs (u, k) where u < k?
                        # The C4 is u - w1 - k - w2 - u.
                        # This is covered by checking common neighbors of u and k?
                        # No, the constraint is on pairs of vertices (a,b).
                        # When we add edge (k,v) with color c, we might create a C4 involving k and v.
                        # The check `common = adj_mask[c][u] & new_mask_v` checks common neighbors of u and v.
                        # If u is a neighbor of v in color c, we check if u and v share >1 common neighbor.
                        # This is correct.
                        
                        # Do we need to check C4s not involving v? No, only new edges involve v.
                        # Do we need to check C4s involving k and some other w?
                        # Edge (k,v) is new. Any C4 using (k,v) must use v.
                        # So checking pairs (u,v) where (u,v) is color c is sufficient.
                        
                        # Recurse
                        current_adj_mask_v[c] |= (1 << k)
                        if backtrack(k + 1, current_adj_mask_v):
                            return True
                        current_adj_mask_v[c] ^= (1 << k)
                    
                    return False
                
                # Initialize for vertex v
                # We need to pass a copy of adj_mask? No, we update in place on success.
                # But backtrack modifies current_adj_mask_v.
                # We must ensure adj_mask is only updated on success.
                # The `adj_mask` in the closure is the global one.
                # We only update it in the `if k == v` block.
                
                # However, we need to handle the case where we fail and need to backtrack.
                # The `adj_mask` is not modified during the search, only `current_adj_mask_v`.
                # So it's safe.
                
                current_adj_mask_v = [0] * 5
                if backtrack(0, current_adj_mask_v):
                    return True
                return False

            # Start solving
            # We need to reset adj_mask and colors for each attempt
            adj_mask = [[0] * n for _ in range(5)]
            colors = [0] * (n * (n - 1) // 2)
            partial_colors = [0] * n
            
            # Process vertices 0 to n-1
            failed = False
            for v in range(n):
                if not solve_vertex(v, partial_colors, adj_mask):
                    failed = True
                    break
                # partial_colors for v is now set for edges (0,v)...(v-1,v)
                # We don't need to keep it, it's stored in `colors`
            
            if not failed:
                # Verify solution
                # Quick check
                valid = True
                for c in range(5):
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj_mask[c][u] & adj_mask[c][v]).bit_count() > 1:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_n = n
                    best_colors = colors[:]
                    return {'n': best_n, 'colors': best_colors}
        
        # If we couldn't solve n, try n-1
        # But maybe a different random seed for the same n would work?
        # We did 50 attempts. If n=26 is hard, we might waste time.
        # Let's limit attempts based on n.
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
