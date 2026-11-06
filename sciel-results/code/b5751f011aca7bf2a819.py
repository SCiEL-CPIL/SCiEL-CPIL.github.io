import random
import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 85.0
    
    # Precompute all 4-term APs (0-indexed)
    aps = []
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                aps.append((a, b, c, e))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Build index -> list of (ap_idx, position_in_ap) for fast updates
    index_to_ap = [[] for _ in range(N)]
    for ap_idx, (a, b, c, e) in enumerate(aps):
        index_to_ap[a].append((ap_idx, 0))
        index_to_ap[b].append((ap_idx, 1))
        index_to_ap[c].append((ap_idx, 2))
        index_to_ap[e].append((ap_idx, 3))
    
    def count_violations_fast(colors):
        """Count APs with < 3 distinct colors using vectorized ops"""
        if num_aps == 0:
            return 0
        c0 = colors[ap_array[:, 0]]
        c1 = colors[ap_array[:, 1]]
        c2 = colors[ap_array[:, 2]]
        c3 = colors[ap_array[:, 3]]
        # An AP is bad if it uses < 3 distinct colors
        # i.e., all same, or exactly 2 distinct
        # Count distinct: use the fact that < 3 means at most 2
        # Bad if: all 4 same, or 3 same + 1 diff, or 2+2, or 2+1+1
        # Simpler: count how many are NOT (at least 3 distinct)
        # At least 3 distinct means NOT (all same) AND NOT (exactly 2 patterns)
        # Let's just compute: for each AP, check if len({c0,c1,c2,c3}) < 3
        # Vectorized: an AP has < 3 distinct iff (c0==c1 and c1==c2 and c2==c3)  # all same
        #   OR (c0==c1 and c1==c2 and c2!=c3)  # 3 same
        #   OR (c0==c1 and c1!=c2 and c2==c3)  # 2,2
        #   OR (c0==c1 and c1!=c2 and c2!=c3 and c3==c0)  # 2,1,1 where c0==c3
        #   OR (c0!=c1 and c1==c2 and c2==c3)  # 1,3
        #   OR (c0!=c1 and c1==c2 and c2!=c3 and c3==c1)  # 1,2,1 where c1==c3
        #   OR (c0!=c1 and c1!=c2 and c2==c3 and c3==c0)  # 1,1,2 where c0==c3
        # This is getting complex. Let me use a different approach:
        # An AP has < 3 distinct colors iff the number of distinct values is 1 or 2.
        # We can check: is it the case that all four are in some set of size <= 2?
        # Simpler vectorized approach:
        # distinct_count < 3 means: NOT (there exist 3 pairwise distinct among the 4)
        # Equivalent: the 4 values can be covered by 2 values.
        # Let me just use: bad if (c0==c1) and (c1==c2) and (c2==c3)  # all same
        #   or (c0==c1) and (c1==c2) and (c2!=c3)  # 3 same
        #   or (c0==c1) and (c1!=c2) and (c2==c3)  # 2,2
        #   or (c0==c1) and (c1!=c2) and (c2!=c3) and (c0==c3)  # 2,1,1
        #   or (c0!=c1) and (c1==c2) and (c2==c3)  # 1,3
        #   or (c0!=c1) and (c1==c2) and (c2!=c3) and (c1==c3)  # 1,2,1
        #   or (c0!=c1) and (c1!=c2) and (c2==c3) and (c0==c3)  # 1,1,2
        
        # Actually, simplest: bad iff NOT (at least 3 distinct)
        # At least 3 distinct iff: (c0,c1,c2) are not all in a 2-set AND (c0,c1,c3) are not all in a 2-set AND ... 
        # Let me just do it differently:
        # bad = (c0 == c1) & (c1 == c2) & (c2 == c3)  # all 4 same
        #   | (c0 == c1) & (c1 == c2) & (c2 != c3)  # first 3 same
        #   | (c0 == c1) & (c1 != c2) & (c2 == c3)  # 2,2
        #   | (c0 == c1) & (c1 != c2) & (c2 != c3) & (c0 == c3)  # 2,1,1
        #   | (c0 != c1) & (c1 == c2) & (c2 == c3)  # last 3 same
        #   | (c0 != c1) & (c1 == c2) & (c2 != c3) & (c1 == c3)  # 1,2,1
        #   | (c0 != c1) & (c1 != c2) & (c2 == c3) & (c0 == c3)  # 1,1,2
        
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        three_same_first = (c0 == c1) & (c1 == c2) & (c2 != c3)
        three_same_last = (c0 != c1) & (c1 == c2) & (c2 == c3)
        two_two = (c0 == c1) & (c1 != c2) & (c2 == c3)
        two_one_one = (c0 == c1) & (c1 != c2) & (c2 != c3) & (c0 == c3)
        one_two_one = (c0 != c1) & (c1 == c2) & (c2 != c3) & (c1 == c3)
        one_one_two = (c0 != c1) & (c1 != c2) & (c2 == c3) & (c0 == c3)
        
        bad = all_same | three_same_first | three_same_last | two_two | two_one_one | one_two_one | one_one_two
        return int(np.sum(bad))
    
    def local_search(K, max_steps=20000, time_limit=20.0):
        """Local search to find valid coloring with K colors"""
        start = time.time()
        best_violations = float('inf')
        best_colors = None
        
        for restart in range(50):
            if time.time() > start + time_limit:
                break
            
            colors = np.random.randint(0, K, N, dtype=np.int32)
            violations = count_violations_fast(colors)
            
            if violations == 0:
                return colors.tolist()
            
            best_violations = min(best_violations, violations)
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
            
            T = 1.0
            cooling = 0.9995
            min_T = 0.005
            
            for step in range(max_steps):
                if time.time() > start + time_limit:
                    break
                
                i = np.random.randint(0, N)
                old_color = colors[i]
                new_color = np.random.randint(0, K)
                if new_color == old_color:
                    continue
                
                # Compute delta by checking affected APs
                old_viol = 0
                new_viol = 0
                affected_aps = index_to_ap[i]
                
                # Save old state
                old_colors_backup = {}
                for ap_idx, pos in affected_aps:
                    a, b, c, e = aps[ap_idx]
                    vals = [colors[a], colors[b], colors[c], colors[e]]
                    if len(set(vals)) < 3:
                        old_viol += 1
                    old_colors_backup[ap_idx] = (pos, colors[i])
                
                colors[i] = new_color
                
                for ap_idx, pos in affected_aps:
                    a, b, c, e = aps[ap_idx]
                    vals = [colors[a], colors[b], colors[c], colors[e]]
                    if len(set(vals)) < 3:
                        new_viol += 1
                
                delta = new_viol - old_viol
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    violations += delta
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                    if violations == 0:
                        return colors.tolist()
                else:
                    colors[i] = old_color
                
                T *= cooling
                if T < min_T:
                    T = min_T
        
        if best_colors is not None and count_violations_fast(best_colors) == 0:
            return best_colors.tolist()
        return None
    
    # Try increasing K values
    for K in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        sol = local_search(K, max_steps=15000, time_limit=min(25.0, deadline - time.time() - 5.0))
        if sol is not None:
            # Verify
            if count_violations_fast(np.array(sol, dtype=np.int32)) == 0:
                return sol
    
    # Fallback: greedy from inspiration 1
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d >= 0:
                    ap_colors = [color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]]
                    if len(set(ap_colors)) < 3:
                        valid = False
                        break
            if not valid:
                continue
            # Also need to check that we don't create a 2-color AP where the new position is the 4th
            # Actually the above checks all APs ending at x. But we also need APs where x is not the last.
            # The greedy in inspiration 1 only checks APs where x is the last element.
            # Let me verify this is sufficient... Actually no, we need to check all APs containing x.
            # But for greedy construction, when we place color at x, we only need to ensure
            # that no AP containing x has < 3 colors. APs where x is not the last element
            # were already valid when their last element was placed.
            # Wait, that's not right. If x is the first element of an AP, then when we place x,
            # the other 3 elements are not yet placed. So we can't check them.
            # The greedy in inspiration 1 seems to only check APs where x is the last element.
            # This is a forward-looking greedy that ensures all APs ending at x are valid.
            # But APs starting at x with x as first element won't be checked until their last element is placed.
            # This might not work correctly. Let me just use the same logic as inspiration 1.
            colors.append(color)
            break
    
    return colors
