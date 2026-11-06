import time
import random
import numpy as np
from collections import defaultdict, Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs and index them by position for fast updates
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)
    
    def count_violations_full(colors):
        """
        Full vectorized count of violations.
        Returns number of APs with < 3 distinct colors.
        """
        colors_np = np.array(colors, dtype=np.int32)
        ap_colors = colors_np[aps_np]
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        
        # Distinct count < 3 means:
        # 1. All 4 same
        # 2. 3 same, 1 diff
        # 3. 2 pairs (e.g., A A B B, A B A B, A B B A)
        
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Exactly 2 distinct colors
        # Case 1: c0==c1 and c2==c3 and c0!=c2 (AABB)
        # Case 2: c0==c2 and c1==c3 and c0!=c1 (ABAB)
        # Case 3: c0==c3 and c1==c2 and c0!=c1 (ABBA)
        # Case 4: c0==c1 and c1!=c2 and c2==c3 -> covered by 1? No.
        # Let's use a simpler logic:
        # If len({c0,c1,c2,c3}) < 3
        
        # Vectorized distinct count is tricky. 
        # Alternative: 
        # Violation if:
        # (c0==c1 and c1==c2 and c2==c3) # 4 same
        # OR (c0==c1 and c1==c2 and c2!=c3) # 3 same (0,1,2)
        # OR (c0==c1 and c1!=c2 and c2==c3) # 3 same (1,2,3) ? No, 0,1 same, 2,3 same. If 0!=2, it's 2 pairs.
        
        # Let's just check if number of distinct is < 3.
        # Distinct >= 3 is equivalent to NOT (all same OR exactly 2 distinct).
        
        # Exactly 2 distinct:
        # Pairs:
        # 01-23: c0==c1, c2==c3, c0!=c2
        # 02-13: c0==c2, c1==c3, c0!=c1
        # 03-12: c0==c3, c1==c2, c0!=c1
        # Triples:
        # 012-3: c0==c1, c1==c2, c0!=c3
        # 013-2: c0==c1, c1==c3, c0!=c2
        # 023-1: c0==c2, c2==c3, c0!=c1
        # 123-0: c1==c2, c2==c3, c1!=c0
        
        v = all_same | \
            ((c0 == c1) & (c2 == c3) & (c0 != c2)) | \
            ((c0 == c2) & (c1 == c3) & (c0 != c1)) | \
            ((c0 == c3) & (c1 == c2) & (c0 != c1)) | \
            ((c0 == c1) & (c1 == c2) & (c0 != c3)) | \
            ((c0 == c1) & (c1 == c3) & (c0 != c2)) | \
            ((c0 == c2) & (c2 == c3) & (c0 != c1)) | \
            ((c1 == c2) & (c2 == c3) & (c1 != c0))
            
        return int(np.sum(v))

    def get_initial_coloring():
        """
        Greedy construction to get a valid coloring with small K.
        """
        colors = []
        for x in range(1, N + 1):
            # Try colors 0, 1, 2, ...
            found = False
            for color in range(10): # Try first 10 colors
                valid = True
                # Check all 4-APs ending at x
                # x = a + 3d => a = x - 3d
                # x = a + 2d => a = x - 2d
                # x = a + d  => a = x - d
                # x = a      => no AP ending here with x as 4th term? 
                # Wait, the AP is (a, a+d, a+2d, a+3d).
                # If x is the 4th term: a+3d=x.
                # If x is the 3rd term: a+2d=x.
                # If x is the 2nd term: a+d=x.
                # If x is the 1st term: a=x.
                
                # We need to check all APs that include x.
                # It's easier to just check all APs that include x using precomputed indices.
                # But we are building incrementally. 
                # Let's just check all APs that include x and have all other positions < x.
                
                for ap_idx in ap_indices[x-1]: # x-1 is 0-indexed
                    ap = aps[ap_idx]
                    # Check if all other positions are < x (i.e., already colored)
                    if all(pos < x-1 for pos in ap if pos != x-1):
                        # Get colors of other positions
                        other_colors = [colors[pos] for pos in ap if pos != x-1]
                        new_set = set(other_colors + [color])
                        if len(new_set) < 3:
                            valid = False
                            break
                
                if valid:
                    colors.append(color)
                    found = True
                    break
            
            if not found:
                # Fallback: try more colors
                for color in range(10, 20):
                    valid = True
                    for ap_idx in ap_indices[x-1]:
                        ap = aps[ap_idx]
                        if all(pos < x-1 for pos in ap if pos != x-1):
                            other_colors = [colors[pos] for pos in ap if pos != x-1]
                            new_set = set(other_colors + [color])
                            if len(new_set) < 3:
                                valid = False
                                break
                    if valid:
                        colors.append(color)
                        break
        
        return colors

    def local_search(colors, target_k, time_limit):
        """
        Local search to minimize violations for a fixed target_k.
        Returns (colors, violations).
        """
        colors = list(colors)
        # Remap colors to 0..target_k-1
        unique_colors = sorted(set(colors))
        if len(unique_colors) > target_k:
            # Map extra colors randomly
            color_map = {c: i % target_k for i, c in enumerate(unique_colors)}
            # Better: map first target_k to 0..k-1, rest random
            for i, c in enumerate(unique_colors):
                if i < target_k:
                    color_map[c] = i
                else:
                    color_map[c] = random.randint(0, target_k - 1)
            colors = [color_map[c] for c in colors]
        else:
            # Ensure we use colors in 0..k-1
            color_map = {c: i for i, c in enumerate(unique_colors)}
            # Fill unused colors? No, just use the map.
            # If we have fewer colors than target_k, that's fine, but we want to use up to target_k.
            # Actually, we just need a valid coloring using AT MOST target_k colors.
            # The remapping above ensures colors are in 0..target_k-1.
            pass

        # Initialize violation flags
        v_flags = np.zeros(num_aps, dtype=bool)
        for ai in range(num_aps):
            ap = aps_np[ai]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            v_flags[ai] = len({c0, c1, c2, c3}) < 3
        
        num_violations = int(np.sum(v_flags))
        
        if num_violations == 0:
            return colors, 0
        
        best_colors = colors[:]
        best_violations = num_violations
        temperature = 1.0
        cooling_rate = 0.9995
        
        while time.time() < time_limit:
            if num_violations == 0:
                return colors, 0
            
            if num_violations < best_violations:
                best_violations = num_violations
                best_colors = colors[:]
                temperature = max(temperature * 1.1, 1.0) # Reheat slightly on improvement? Or just keep cooling.
            
            # Pick a random violated AP
            violated_idx = np.where(v_flags)[0]
            if len(violated_idx) == 0:
                break
            ai = int(random.choice(violated_idx))
            ap = aps_np[ai]
            
            # Pick a random position in the AP to change
            pos_in_ap = random.randint(0, 3)
            idx = int(ap[pos_in_ap])
            old_c = colors[idx]
            
            # Try a new color
            new_c = random.randint(0, target_k - 1)
            if new_c == old_c:
                continue
            
            # Calculate delta
            delta = 0
            old_flags = []
            for aai in ap_indices[idx]:
                old_flag = bool(v_flags[aai])
                old_flags.append((aai, old_flag))
                
                ap2 = aps_np[aai]
                # Get colors, replacing idx with new_c
                c0 = new_c if ap2[0] == idx else colors[ap2[0]]
                c1 = new_c if ap2[1] == idx else colors[ap2[1]]
                c2 = new_c if ap2[2] == idx else colors[ap2[2]]
                c3 = new_c if ap2[3] == idx else colors[ap2[3]]
                
                is_violated = len({c0, c1, c2, c3}) < 3
                
                if old_flag and not is_violated:
                    delta -= 1
                elif not old_flag and is_violated:
                    delta += 1
            
            # Acceptance
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < np.exp(-delta / max(temperature, 0.1))
            
            if accept:
                colors[idx] = new_c
                for aai, old_flag in old_flags:
                    # Recalculate flag for aai
                    ap2 = aps_np[aai]
                    c0 = colors[ap2[0]]
                    c1 = colors[ap2[1]]
                    c2 = colors[ap2[2]]
                    c3 = colors[ap2[3]]
                    v_flags[aai] = len({c0, c1, c2, c3}) < 3
                num_violations += delta
                temperature *= cooling_rate
            else:
                # Revert not needed as we didn't change colors[idx]
                pass
            
            if temperature < 0.01:
                temperature = 1.0 # Reset if too cold? Or break?
                # Let's just keep it low.

        if best_violations < num_violations:
            return best_colors, best_violations
        return colors, num_violations

    # Main Logic
    # 1. Get a valid coloring with K=8
    init_colors = get_initial_coloring()
    current_k = len(set(init_colors))
    best_colors = init_colors
    best_k = current_k
    
    # 2. Try to reduce K
    # We check K from 3 to best_k - 1
    # Allocate time for each K
    # If best_k is 8, we try 3, 4, 5, 6, 7
    
    # Sort Ks to try
    k_to_try = list(range(3, best_k))
    
    time_per_k = (deadline - time.time()) / (len(k_to_try) + 1)
    
    for target_k in k_to_try:
        if time.time() > deadline - 5:
            break
        
        # Remap current best colors to target_k
        # Use the best colors found so far as the starting point for remapping
        # This is better than random init
        
        # We need to be careful: if best_k > target_k, we remap.
        # If best_k <= target_k, we don't need to remap, but we might have unused colors.
        
        # Let's just run local search on a remapped version of best_colors
        # To encourage exploration, we might also try random restarts for small K
        
        time_limit = min(deadline - 5, time.time() + time_per_k)
        
        # Try multiple restarts for this K
        found_valid = False
        for restart in range(3):
            if time.time() > time_limit:
                break
            
            if restart == 0:
                # Start from remapped best
                init_for_k = best_colors[:]
            else:
                # Random start
                init_for_k = [random.randint(0, target_k - 1) for _ in range(N)]
            
            res_colors, res_violations = local_search(init_for_k, target_k, time_limit)
            
            if res_violations == 0:
                # Valid coloring with target_k colors
                best_colors = res_colors
                best_k = target_k
                found_valid = True
                break
        
        if found_valid:
            # We found a valid coloring with target_k.
            # Since we are iterating from small K to large K, this is the minimum K so far.
            # We can stop if we found K=3.
            if target_k == 3:
                break
            # If we found a smaller K, we should continue to see if we can find an even smaller one?
            # No, we are iterating 3, 4, 5...
            # If we found 4, we should have already tried 3.
            # If 3 failed, 4 is the best.
            # So if we find a valid one, it is the minimum possible (given our search order).
            # Wait, if we try 3, 4, 5... and 3 fails, 4 succeeds, then 4 is the min.
            # We can break after the first success because we are going up.
            break

    return best_colors
# EVOLVE-BLOCK-END
