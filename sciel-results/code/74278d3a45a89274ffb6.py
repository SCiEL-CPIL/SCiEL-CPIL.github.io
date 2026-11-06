import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Fallback: n=11 construction from seed
    # This is a known valid coloring for n=11
    def get_seed_colors(n):
        # Simple cyclic coloring often works for small n
        # For n=11, colors = (j-i) % 5 might not work if it creates C4s.
        # Let's use a known valid one or just return random if n is small.
        # Actually, the seed program used:
        # [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
        # This is a specific pattern. Let's reuse it for n=11.
        if n == 11:
            return [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
        # For other n, we will rely on search
        return [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]

    best_n = 11
    best_colors = get_seed_colors(11)
    
    # Try to find larger n
    # We will iterate n from 12 upwards
    # We keep the best solution found so far
    
    for n in range(12, 27):
        if time.time() > deadline - 5.0:
            break
            
        # Time budget for this n
        # Allocate more time for larger n if we have time, but ensure we don't run out
        # If we are at n=20 and have 50s left, we can use it.
        # Let's just run until deadline - 2.0
        
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge map
        edge_map = {}
        inv_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                inv_map[idx] = (i, j)
                idx += 1
        
        # adj_mask[c][v] is a bitmask of neighbors of v in color c
        adj_mask = [[0] * n for _ in range(5)]
        
        def update_mask(c, i, j, add):
            if add:
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            else:
                adj_mask[c][i] &= ~(1 << j)
                adj_mask[c][j] &= ~(1 << i)
                
        # Initialize masks and violations
        # Violation count: sum over all pairs (u,v) of max(0, common_neighbors(u,v) - 1)
        # We maintain this incrementally.
        
        current_violations = 0
        
        # Initial setup
        for e_idx in range(num_edges):
            i, j = inv_map[e_idx]
            c = colors[e_idx]
            update_mask(c, i, j, True)
            
        # Calculate initial violations
        def calc_initial_violations():
            v = 0
            for c in range(5):
                for u in range(n):
                    for vrt in range(u + 1, n):
                        common = adj_mask[c][u] & adj_mask[c][vrt]
                        cnt = common.bit_count()
                        if cnt > 1:
                            v += cnt - 1
            return v
            
        current_violations = calc_initial_violations()
        
        # Local Search
        temp = 1.0
        cooling = 0.9999
        max_iters = 500000
        
        # To speed up, we can precompute list of edges? No, random is fine.
        # We need to efficiently calculate delta.
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_violations == 0:
                best_n = n
                best_colors = colors[:]
                break
                
            e_idx = random.randint(0, num_edges - 1)
            i, j = inv_map[e_idx]
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            delta = 0
            
            # Helper to get common neighbors count
            def get_common_count(c, u, v):
                return (adj_mask[c][u] & adj_mask[c][v]).bit_count()
                
            # 1. Remove edge (i,j) from old_color
            c = old_color
            # Affected pairs: (i, w) for w in N_c(j) \ {i}, (j, w) for w in N_c(i) \ {j}
            # Note: (i,j) itself is not a pair in the violation sum (u<v), 
            # but removing it affects common neighbors of (i,w) and (j,w).
            
            mask_j = adj_mask[c][j]
            # Iterate over neighbors of j in color c
            temp_mask = mask_j
            while temp_mask:
                lsb = temp_mask & -temp_mask
                w = lsb.bit_length() - 1
                temp_mask ^= lsb
                if w == i: continue
                
                # Pair (i, w)
                # Before removal: common(i, w) includes j if j is neighbor of i (yes) and w (yes, w is in N(j) so j is in N(w)).
                # Actually, common(i, w) counts x such that x in N(i) and x in N(w).
                # j is in N(i) (edge i-j exists).
                # j is in N(w) (edge j-w exists).
                # So j is a common neighbor of i and w.
                # When we remove edge i-j, j is no longer in N(i).
                # So common(i, w) decreases by 1.
                
                n_old = get_common_count(c, i, w)
                n_new = n_old - 1
                delta += max(0, n_new - 1) - max(0, n_old - 1)
                
                # Pair (j, w)
                # j is in N(w) (yes).
                # i is in N(j) (yes, being removed).
                # Is i in N(w)?
                # If i is in N(w), then i is a common neighbor of j and w.
                # When we remove i-j, i is no longer in N(j).
                # So if i was a common neighbor, it stops being one.
                if (adj_mask[c][w] >> i) & 1:
                    n_old_jw = get_common_count(c, j, w)
                    n_new_jw = n_old_jw - 1
                    delta += max(0, n_new_jw - 1) - max(0, n_old_jw - 1)
            
            # Update masks for old_color
            update_mask(c, i, j, False)
            
            # 2. Add edge (i,j) to new_color
            c = new_color
            mask_j = adj_mask[c][j]
            temp_mask = mask_j
            while temp_mask:
                lsb = temp_mask & -temp_mask
                w = lsb.bit_length() - 1
                temp_mask ^= lsb
                if w == i: continue
                
                # Pair (i, w)
                # j becomes a neighbor of i.
                # j is already a neighbor of w (since w in N(j)).
                # So j becomes a common neighbor of i and w.
                n_old = get_common_count(c, i, w)
                n_new = n_old + 1
                delta += max(0, n_new - 1) - max(0, n_old - 1)
                
                # Pair (j, w)
                # i becomes a neighbor of j.
                # If i is a neighbor of w, then i becomes a common neighbor of j and w.
                if (adj_mask[c][w] >> i) & 1:
                    n_old_jw = get_common_count(c, j, w)
                    n_new_jw = n_old_jw + 1
                    delta += max(0, n_new_jw - 1) - max(0, n_old_jw - 1)
            
            # Update masks for new_color
            update_mask(c, i, j, True)
            
            new_violations = current_violations + delta
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                colors[e_idx] = new_color
            else:
                # Revert
                update_mask(new_color, i, j, False)
                update_mask(old_color, i, j, True)
                
            temp *= cooling
            
        if current_violations == 0:
            best_n = n
            best_colors = colors[:]
        else:
            # If we failed, we might still have a good solution for previous n.
            # We continue to next n only if we have time.
            # But usually if we fail at n, we fail at n+1.
            # However, random restarts might work.
            # Let's break to save time for verification or smaller n if we didn't update best_n?
            # No, best_n is already set.
            # If we just failed n=12, we should probably try n=12 again with different seed?
            # Or break.
            # Given time constraints, breaking is safer to return best_n found so far.
            break
            
    return {'n': best_n, 'colors': best_colors}
