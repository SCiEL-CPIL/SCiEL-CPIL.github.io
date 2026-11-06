import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Fallback: n=11
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Helper: Index of edge (i, j) in lexicographic list
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Helper: Check if a coloring is valid
    def check_validity(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        if common.bit_count() > 1:
                            return False
        return True

    # Helper: Count violations (C4s) using the sum of binom(common, 2)
    def count_violations(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt * (cnt - 1) // 2
        return violations

    # Helper: Build adjacency masks from colors
    def build_masks(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        return adj_mask

    # Local Search with Incremental Violation Counting
    def local_search_incremental(colors, n, max_time):
        adj_mask = build_masks(colors, n)
        
        # Precompute initial violations
        current_violations = count_violations(colors, n)
        
        if current_violations == 0:
            return colors, True
        
        best_colors = colors[:]
        best_violations = current_violations
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        
        # Simulated Annealing
        temp = 5.0
        cooling = 0.999
        min_temp = 0.01
        
        # Helper to calculate delta for changing edge (u, v) from c_old to c_new
        def calc_delta(u, v, c_old, c_new):
            if c_old == c_new:
                return 0
            
            delta = 0
            
            # 1. Effects in c_old (Removal)
            # Pairs involving u or v
            # For each w != u, v:
            #   If (u, w) is in c_old and (v, w) is in c_old:
            #      Common neighbors of (u, w) decreases by 1 (v is removed from N(u))
            #      Common neighbors of (v, w) decreases by 1 (u is removed from N(v))
            #   Wait, let's trace carefully.
            #   Change: Remove (u, v) from c_old.
            #   N_c_old(u) loses v. N_c_old(v) loses u.
            
            # Pair (u, w) for w != u, v:
            #   Old common: |N(u) \cap N(w)|
            #   New common: |(N(u) \ {v}) \cap N(w)| = |N(u) \cap N(w)| - 1 if v in N(w), else |N(u) \cap N(w)|
            #   So if (v, w) in c_old, k decreases by 1.
            
            # Pair (v, w) for w != u, v:
            #   Old common: |N(v) \cap N(w)|
            #   New common: |(N(v) \ {u}) \cap N(w)| = |N(v) \cap N(w)| - 1 if u in N(w), else |N(v) \cap N(w)|
            #   So if (u, w) in c_old, k decreases by 1.
            
            # Pair (u, v):
            #   Old common: |N(u) \cap N(v)|
            #   New common: |(N(u) \ {v}) \cap (N(v) \ {u})|
            #   Since u not in N(u) and v not in N(v), and u not in N(u) \cap N(v) (as u not in N(u)),
            #   the intersection is unchanged.
            #   So f(u, v) is unchanged.
            
            # So we iterate w from 0 to n-1, w != u, w != v.
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Check if (u, w) and (v, w) are in c_old
                # We need to know if (u, w) is in c_old and (v, w) is in c_old
                # If yes, then for pair (u, w), common count decreases by 1 (v was common neighbor)
                # And for pair (v, w), common count decreases by 1 (u was common neighbor)
                
                u_w_in = (adj_mask[c_old][u] >> w) & 1
                v_w_in = (adj_mask[c_old][v] >> w) & 1
                
                if u_w_in and v_w_in:
                    # Pair (u, w): k decreases by 1
                    # Old k: |N(u) \cap N(w)|. New k: k-1.
                    # Delta f = C(k-1, 2) - C(k, 2) = (k-1)(k-2)/2 - k(k-1)/2 = (k-1)(k-2-k)/2 = -(k-1)
                    # We need the old k.
                    common_uw = (adj_mask[c_old][u] & adj_mask[c_old][w]).bit_count()
                    # Note: This common_uw includes v? Yes, because v is in N(u) and v is in N(w).
                    # So old k = common_uw.
                    # New k = common_uw - 1.
                    # Delta = - (common_uw - 1)
                    delta -= (common_uw - 1)
                    
                    # Pair (v, w): k decreases by 1
                    # Old k = |N(v) \cap N(w)| = common_uw (symmetric)
                    # New k = common_uw - 1
                    # Delta = - (common_uw - 1)
                    delta -= (common_uw - 1)
                
                elif u_w_in:
                    # (u, w) in c_old, (v, w) not in c_old.
                    # Pair (u, w): v is NOT in N(w), so v was not a common neighbor.
                    # Common count unchanged.
                    # Pair (v, w): u is in N(w)? No, (v, w) not in c_old.
                    # Wait, if (v, w) is not in c_old, then w is not in N(v).
                    # So pair (v, w) is not an edge? No, pair (v, w) refers to the vertices v and w.
                    # The term f(v, w) is defined for all pairs v, w, not just edges.
                    # But if (v, w) is not an edge, do we care?
                    # The sum is over all pairs x < y.
                    # If (v, w) is not an edge, N(v) and N(w) are still defined.
                    # But if (v, w) is not an edge, does changing (u, v) affect f(v, w)?
                    # f(v, w) depends on N(v) \cap N(w).
                    # N(v) loses u.
                    # So if u in N(w), then common count decreases by 1.
                    # u in N(w) means (u, w) in c_old.
                    # We are in the case u_w_in (True).
                    # So u is in N(w).
                    # So for pair (v, w), common count decreases by 1.
                    # Old k = |N(v) \cap N(w)|.
                    # New k = k - 1.
                    # Delta = - (k - 1).
                    common_vw = (adj_mask[c_old][v] & adj_mask[c_old][w]).bit_count()
                    delta -= (common_vw - 1)
                    
                    # What about pair (u, w)?
                    # N(u) loses v.
                    # v in N(w)? No, v_w_in is False.
                    # So v was not a common neighbor of u and w.
                    # Common count unchanged.
                    
                elif v_w_in:
                    # (v, w) in c_old, (u, w) not in c_old.
                    # Symmetric to above.
                    # Pair (u, w): N(u) loses v. v in N(w)? Yes (v_w_in).
                    # So common count of (u, w) decreases by 1.
                    common_uw = (adj_mask[c_old][u] & adj_mask[c_old][w]).bit_count()
                    delta -= (common_uw - 1)
                    
                    # Pair (v, w): N(v) loses u. u in N(w)? No.
                    # Unchanged.

            # 2. Effects in c_new (Addition)
            # Add (u, v) to c_new.
            # N_c_new(u) gains v. N_c_new(v) gains u.
            
            for w in range(n):
                if w == u or w == v:
                    continue
                
                u_w_in = (adj_mask[c_new][u] >> w) & 1
                v_w_in = (adj_mask[c_new][v] >> w) & 1
                
                if u_w_in and v_w_in:
                    # Pair (u, w): v is now in N(u) and v is in N(w).
                    # v becomes a new common neighbor.
                    # Old k = |N(u) \cap N(w)|.
                    # New k = k + 1.
                    # Delta = C(k+1, 2) - C(k, 2) = k.
                    common_uw = (adj_mask[c_new][u] & adj_mask[c_new][w]).bit_count()
                    delta += common_uw
                    
                    # Pair (v, w): u is now in N(v) and u is in N(w).
                    # u becomes a new common neighbor.
                    # Old k = |N(v) \cap N(w)| = common_uw.
                    # New k = k + 1.
                    # Delta = k.
                    delta += common_uw
                
                elif u_w_in:
                    # (u, w) in c_new, (v, w) not in c_new.
                    # Pair (v, w): u is now in N(v). u is in N(w) (since u_w_in).
                    # u becomes a new common neighbor of v and w.
                    # Old k = |N(v) \cap N(w)|.
                    # New k = k + 1.
                    # Delta = k.
                    common_vw = (adj_mask[c_new][v] & adj_mask[c_new][w]).bit_count()
                    delta += common_vw
                    
                    # Pair (u, w): v is now in N(u). v in N(w)? No.
                    # Unchanged.
                    
                elif v_w_in:
                    # Symmetric.
                    # Pair (u, w): v is now in N(u). v is in N(w).
                    # v becomes a new common neighbor.
                    common_uw = (adj_mask[c_new][u] & adj_mask[c_new][w]).bit_count()
                    delta += common_uw

            return delta

        while temp > min_temp and time.time() < max_time:
            e_idx = random.randint(0, num_edges - 1)
            u, v = edges[e_idx]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            delta = calc_delta(u, v, old_color, new_color)
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # Accept
                colors[e_idx] = new_color
                # Update masks
                adj_mask[old_color][u] &= ~(1 << v)
                adj_mask[old_color][v] &= ~(1 << u)
                adj_mask[new_color][u] |= (1 << v)
                adj_mask[new_color][v] |= (1 << u)
                
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Reject
                pass
            
            temp *= cooling
            
        if best_violations == 0:
            return best_colors, True
        return best_colors, False

    # Strategy:
    # 1. Try n=26 with structured init (Slope-based)
    # 2. If fails, try n=25, etc.
    
    # Structured Init for n=26
    # Vertices 0..24 are (x, y) in Z_5 x Z_5. Vertex 25 is "infinity".
    # Color of (x1, y1), (x2, y2):
    # If x1 == x2: slope = infinity.
    # Else: slope = (y2 - y1) * (x2 - x1)^-1 mod 5.
    # Slopes: 0, 1, 2, 3, 4, inf.
    # Map to 5 colors.
    # Merge slope 0 and 1?
    # Let's try merging 0 and inf?
    # If we merge 0 and inf, we get horizontal and vertical lines same color.
    # This creates C4s in the grid.
    # We need a mapping that avoids C4s.
    # Since we don't know the exact mapping, we will use random init but with a "bias" towards structured?
    # No, just use random init for n=26, but the incremental search is fast enough.
    
    n_target = 26
    colors = [random.randint(0, 4) for _ in range(n_target * (n_target - 1) // 2)]
    
    t_limit = time.time() + 30.0
    result_colors, is_valid = local_search_incremental(colors, n_target, t_limit)
    
    if is_valid:
        best_n = n_target
        best_colors = result_colors
    else:
        # Try smaller n
        for n in range(25, 11, -1):
            if time.time() > deadline - 10.0:
                break
            
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            t_limit = time.time() + 10.0
            result_colors, is_valid = local_search_incremental(colors, n, t_limit)
            
            if is_valid:
                best_n = n
                best_colors = result_colors
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
