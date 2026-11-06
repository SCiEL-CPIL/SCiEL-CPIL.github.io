import random
import time

def construct():
    # Try to find a valid coloring for n=26 using local search
    # If it fails, try n=25, etc.
    for n in [26, 25, 24]:
        if time.time() > 110:
            break
        result = solve_local_search(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Fallback to a known small valid coloring
    # n=11 construction from seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_local_search(n, max_iters=500000):
    # colors[i][j] for i < j
    # We store colors in a flat list or 2D array for fast access
    # Let's use a 2D list colors[u][v] for u != v, symmetric
    
    # Initialize random coloring
    colors = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, 4)
            colors[i][j] = c
            colors[j][i] = c
            
    # Neighbor bitmasks for each color
    # N[c][u] is an integer bitmask of neighbors of u in color c
    N = [[0]*n for _ in range(5)]
    
    def build_bitmasks():
        for c in range(5):
            for u in range(n):
                mask = 0
                for v in range(n):
                    if u != v and colors[u][v] == c:
                        mask |= (1 << v)
                N[c][u] = mask
                
    build_bitmasks()
    
    def calc_obj():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    inter = N[c][u] & N[c][v]
                    k = inter.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
        
    current_obj = calc_obj()
    
    # Local search
    temp = 1.0
    min_temp = 0.001
    cooling = 0.9999
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for it in range(max_iters):
        if time.time() > 110:
            break
            
        if current_obj == 0:
            # Extract colors in lexicographic order
            out_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    out_colors.append(colors[i][j])
            return out_colors
            
        # Pick random edge
        u, v = random.choice(edges)
        old_c = colors[u][v]
        
        # Try all new colors
        best_c = old_c
        best_obj = current_obj
        best_delta = 0
        
        for new_c in range(5):
            if new_c == old_c:
                continue
                
            # Simulate change
            # Remove v from N[old_c][u], u from N[old_c][v]
            # Add v to N[new_c][u], u from N[new_c][v]
            
            # Calculate change in objective function
            # Only pairs involving u or v and colors old_c, new_c are affected
            # Plus the pair (u,v) itself
            
            delta = 0
            
            # Helper to get contribution of pair (x,y) for color c
            def contrib(x, y, c):
                inter = N[c][x] & N[c][y]
                k = inter.bit_count()
                return k * (k - 1) // 2
                
            # Pairs involving u: (u, w) for w != u, v
            # Pairs involving v: (v, w) for w != u, v
            # Pair (u, v)
            
            # We need to update N temporarily or calculate delta analytically.
            # Analytical calculation is complex.
            # Let's just update N, calc delta, then revert.
            # But calc_obj is O(n^2). We need incremental update.
            
            # Let's store the current contributions for pairs (u, w) and (v, w) for all w and colors.
            # This is too much state.
            
            # Alternative: Just re-calculate the affected parts.
            # The objective is sum over all pairs (x,y) and colors c of binom(|N_c(x) & N_c(y)|, 2).
            # Changing color of (u,v) from old_c to new_c affects:
            # 1. N[old_c][u] loses v, N[old_c][v] loses u.
            # 2. N[new_c][u] gains v, N[new_c][v] gains u.
            
            # This changes the intersection sizes for:
            # - Pairs (u, w) for all w != u, v.
            # - Pairs (v, w) for all w != u, v.
            # - Pair (u, v).
            
            # And only for colors old_c and new_c.
            
            # Let's compute the delta for these specific pairs.
            
            def get_contrib_pair(x, y, c, Nx, Ny):
                # Nx, Ny are the bitmasks for x, y in color c
                inter = Nx & Ny
                k = inter.bit_count()
                return k * (k - 1) // 2

            # Current contributions for affected pairs
            # We need to be careful: when we change N, the contributions change.
            
            # Let's do a "dry run" update on local copies of the affected masks?
            # No, N is a list of lists. We can modify and revert.
            
            # Save old masks
            old_N_old_c_u = N[old_c][u]
            old_N_old_c_v = N[old_c][v]
            old_N_new_c_u = N[new_c][u]
            old_N_new_c_v = N[new_c][v]
            
            # Apply change
            N[old_c][u] &= ~(1 << v)
            N[old_c][v] &= ~(1 << u)
            N[new_c][u] |= (1 << v)
            N[new_c][v] |= (1 << u)
            
            # Calculate new delta
            # Sum of contributions for (u,w) and (v,w) and (u,v) for colors old_c and new_c
            # Before and After.
            
            # This is still O(n) per new_c, so O(5*n) per move.
            # n=26, so 130 operations per move. 500k moves = 65M ops. Might be slow in Python.
            
            # Let's optimize:
            # Only calculate delta for pairs where the intersection size changes.
            
            # For pair (u, w) in color old_c:
            # N[old_c][u] changed. N[old_c][w] did not.
            # So |N[old_c][u] & N[old_c][w]| changed.
            
            # Let's just compute the total delta by summing over w.
            
            delta = 0
            # Pair (u, v)
            # Before:
            b_uv_old = get_contrib_pair(u, v, old_c, old_N_old_c_u, old_N_old_c_v)
            b_uv_new = get_contrib_pair(u, v, new_c, old_N_new_c_u, old_N_new_c_v)
            # Note: For pair (u,v), the masks used are N[c][u] and N[c][v].
            # In the "Before" state for new_c, N[new_c][u] and N[new_c][v] did NOT contain each other.
            # In the "After" state for new_c, they DO contain each other.
            # So we need to use the "After" masks for the new_c contribution of (u,v).
            # And "Before" masks for old_c contribution of (u,v).
            
            # Wait, the contribution of pair (u,v) to the objective is:
            # binom(|N_c(u) & N_c(v)|, 2)
            # For c=old_c:
            # Before: N[old_c][u] had v, N[old_c][v] had u.
            # After: N[old_c][u] no v, N[old_c][v] no u.
            # For c=new_c:
            # Before: N[new_c][u] no v, N[new_c][v] no u.
            # After: N[new_c][u] has v, N[new_c][v] has u.
            
            # So:
            # Delta_uv = (Contrib_after_old_c(u,v) - Contrib_before_old_c(u,v)) + (Contrib_after_new_c(u,v) - Contrib_before_new_c(u,v))
            
            # Let's compute these.
            # Before old_c: masks had v/u.
            # After old_c: masks lost v/u.
            # Before new_c: masks lacked v/u.
            # After new_c: masks gained v/u.
            
            # We already applied the change to N.
            # So N[old_c][u] is "After". N[old_c][v] is "After".
            # N[new_c][u] is "After". N[new_c][v] is "After".
            
            # We need "Before" values.
            # Before old_c(u,v): use old_N_old_c_u, old_N_old_c_v.
            # Before new_c(u,v): use old_N_new_c_u, old_N_new_c_v.
            
            # After old_c(u,v): use N[old_c][u], N[old_c][v] (current).
            # After new_c(u,v): use N[new_c][u], N[new_c][v] (current).
            
            d_uv_old = get_contrib_pair(u, v, old_c, N[old_c][u], N[old_c][v]) - get_contrib_pair(u, v, old_c, old_N_old_c_u, old_N_old_c_v)
            d_uv_new = get_contrib_pair(u, v, new_c, N[new_c][u], N[new_c][v]) - get_contrib_pair(u, v, new_c, old_N_new_c_u, old_N_new_c_v)
            delta += d_uv_old + d_uv_new
            
            # Now for w != u, v
            for w in range(n):
                if w == u or w == v:
                    continue
                    
                # Color old_c
                # Pair (u, w): N[old_c][u] changed. N[old_c][w] same.
                # Before: old_N_old_c_u & N[old_c][w]
                # After: N[old_c][u] & N[old_c][w]
                b_uw_old = get_contrib_pair(u, w, old_c, old_N_old_c_u, N[old_c][w])
                a_uw_old = get_contrib_pair(u, w, old_c, N[old_c][u], N[old_c][w])
                delta += a_uw_old - b_uw_old
                
                # Pair (v, w): N[old_c][v] changed. N[old_c][w] same.
                b_vw_old = get_contrib_pair(v, w, old_c, old_N_old_c_v, N[old_c][w])
                a_vw_old = get_contrib_pair(v, w, old_c, N[old_c][v], N[old_c][w])
                delta += a_vw_old - b_vw_old
                
                # Color new_c
                # Pair (u, w): N[new_c][u] changed.
                b_uw_new = get_contrib_pair(u, w, new_c, old_N_new_c_u, N[new_c][w])
                a_uw_new = get_contrib_pair(u, w, new_c, N[new_c][u], N[new_c][w])
                delta += a_uw_new - b_uw_new
                
                # Pair (v, w): N[new_c][v] changed.
                b_vw_new = get_contrib_pair(v, w, new_c, old_N_new_c_v, N[new_c][w])
                a_vw_new = get_contrib_pair(v, w, new_c, N[new_c][v], N[new_c][w])
                delta += a_vw_new - b_vw_new
                
            # Revert N
            N[old_c][u] = old_N_old_c_u
            N[old_c][v] = old_N_old_c_v
            N[new_c][u] = old_N_new_c_u
            N[new_c][v] = old_N_new_c_v
            
            if delta < best_delta:
                best_delta = delta
                best_c = new_c
                best_obj = current_obj + delta
                
        # Accept best move or random move with probability
        if best_c != old_c:
            # Apply best_c
            colors[u][v] = best_c
            colors[v][u] = best_c
            N[old_c][u] &= ~(1 << v)
            N[old_c][v] &= ~(1 << u)
            N[best_c][u] |= (1 << v)
            N[best_c][v] |= (1 << u)
            current_obj += best_delta
        else:
            # Try random move if stuck?
            pass
            
        temp *= cooling
        
    return None
