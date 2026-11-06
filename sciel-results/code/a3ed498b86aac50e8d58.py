import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions
    # Positions are 0-indexed in the array, but task says 1..128
    # AP: a, a+d, a+2d, a+3d
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < n:
                aps.append((a, a + d, a + 2 * d, i3))
    
    num_aps = len(aps)
    # Store as numpy array for fast access
    aps_arr = np.array(aps, dtype=np.int32)
    
    # Map position to AP indices for fast delta calculation
    # pos_to_aps[pos] = list of AP indices
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    def count_violations(colors, k):
        """
        Vectorized violation count.
        A 4-AP is invalid if it uses < 3 distinct colors.
        This is equivalent to one color appearing >= 3 times.
        """
        if num_aps == 0:
            return 0
        
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        
        invalid_mask = np.zeros(num_aps, dtype=bool)
        
        for c_val in range(k):
            # Count occurrences of c_val in each AP
            cnt = (c0 == c_val) + (c1 == c_val) + (c2 == c_val) + (c3 == c_val)
            # If count >= 3, the AP uses at most 2 distinct colors (since total is 4)
            # e.g. 3 of one, 1 of another -> 2 distinct. 4 of one -> 1 distinct.
            # If count == 2, it could be 2-2 (2 distinct) or 2-1-1 (3 distinct).
            # So we only flag if count >= 3.
            invalid_mask |= (cnt >= 3)
            
        return np.sum(invalid_mask)

    def try_local_search(k, init_colors, seed):
        rng = random.Random(seed)
        colors = np.array(init_colors, dtype=np.int32)
        current_violations = count_violations(colors, k)
        
        if current_violations == 0:
            return colors, 0
            
        best_colors = colors.copy()
        best_violations = current_violations
        
        temp = 1.0
        cooling = 0.9999
        max_steps = 50000
        
        steps = 0
        while steps < max_steps and time.time() < deadline - 5.0:
            steps += 1
            
            # Pick a random position
            i = rng.randint(0, n - 1)
            old_color = colors[i]
            
            # Pick a new color
            new_color = rng.randint(0, k - 1)
            if new_color == old_color:
                continue
            
            # Calculate delta
            # We only need to check APs involving position i
            # For each such AP, check if it was invalid and is now valid, or vice versa
            
            delta = 0
            affected_aps = pos_to_aps[i]
            
            # Optimization: Only check if the color change actually affects the "count >= 3" logic
            # We can just re-evaluate the specific APs
            
            for ap_idx in affected_aps:
                ap = aps_arr[ap_idx]
                # Extract colors for this AP
                c0 = colors[ap[0]]
                c1 = colors[ap[1]]
                c2 = colors[ap[2]]
                c3 = colors[ap[3]]
                
                # Check if invalid BEFORE change
                was_invalid = False
                for cv in range(k):
                    cnt = (c0 == cv) + (c1 == cv) + (c2 == cv) + (c3 == cv)
                    if cnt >= 3:
                        was_invalid = True
                        break
                
                # Apply change
                # Note: colors[i] is still old_color in the array? 
                # We are simulating. 
                # It's easier to just swap in the array, check, swap back? 
                # Or calculate manually.
                # Let's swap in the array for simplicity in this small loop
                colors[i] = new_color
                
                c0 = colors[ap[0]]
                c1 = colors[ap[1]]
                c2 = colors[ap[2]]
                c3 = colors[ap[3]]
                
                is_invalid_now = False
                for cv in range(k):
                    cnt = (c0 == cv) + (c1 == cv) + (c2 == cv) + (c3 == cv)
                    if cnt >= 3:
                        is_invalid_now = True
                        break
                
                if was_invalid and not is_invalid_now:
                    delta -= 1
                elif not was_invalid and is_invalid_now:
                    delta += 1
                
                # Swap back for next AP in loop? 
                # No, we keep the change for the rest of the AP checks if we accept?
                # Standard Simulated Annealing: calculate delta for the whole move, then accept/reject.
                # So we should revert after checking all affected APs.
            
            colors[i] = old_color # Revert before decision
            
            new_violations = current_violations + delta
            
            if new_violations < current_violations or rng.random() < np.exp(-delta / temp):
                colors[i] = new_color
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
                
                if current_violations == 0:
                    return colors, 0
            
            temp *= cooling
            
        return best_colors, best_violations

    # Phase 1: Try to find a valid 3-coloring
    # Try several random seeds and structured starts
    
    best_solution = None
    best_count = 128
    
    # Structured starts for K=3
    starts = []
    # Random
    for s in range(50):
        starts.append([random.randint(0, 2) for _ in range(n)])
    # Pattern i % 3
    starts.append([i % 3 for i in range(n)])
    starts.append([(i // 2) % 3 for i in range(n)])
    # Legendre symbol mod 127 (0, 1, 2)
    # Legendre symbol (x/p) is 1 if x is quadratic residue, -1 if not, 0 if x=0
    # Map 1->0, -1->1, 0->2
    p = 127
    def legendre(a, p):
        if a == 0: return 0
        return pow(a, (p-1)//2, p) # returns 1 or p-1 (which is -1)
    
    leg_colors = []
    for i in range(1, n+1):
        val = legendre(i, p)
        if val == 1:
            leg_colors.append(0)
        elif val == p-1:
            leg_colors.append(1)
        else:
            leg_colors.append(2)
    starts.append(leg_colors)
    
    for k in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
            
        found_valid = False
        for s_idx, start in enumerate(starts):
            if time.time() > deadline - 10:
                break
                
            # If start has more than k colors, skip or clamp
            if k == 3:
                init = [c % 3 for c in start]
            else:
                init = [c % k for c in start]
                
            colors, viol = try_local_search(k, init, seed=s_idx)
            
            if viol == 0:
                # Valid coloring found
                # Minimize distinct colors actually used
                used = set(colors)
                count = len(used)
                if count < best_count:
                    best_count = count
                    # Remap colors to 0..count-1
                    mapping = {c: i for i, c in enumerate(sorted(used))}
                    best_solution = [mapping[c] for c in colors]
                    found_valid = True
                    if count == 3:
                        # Can't do better than 3
                        return best_solution
                    break # Move to next K? No, we want min count. 
                         # If we found 3, we return. If we found 4, we might find 3 later?
                         # But K=3 search is already done if we are at K=4.
                         # So if K=3 failed, min is >= 4.
        
        if found_valid and best_count == 3:
            return best_solution
            
        # If we are at K=3 and didn't find 0 violations, we might still have a good best_solution from K=4?
        # No, we iterate K increasing.
        
    # If no valid solution found with K<=5, fallback to greedy
    if best_solution is None:
        # Greedy construction
        colors = [0] * n
        for i in range(n):
            for c in range(128):
                valid = True
                # Check all APs ending at i or including i? 
                # Greedy usually checks constraints involving i where other positions are fixed.
                # Since we go left to right, we check APs where i is the last element?
                # No, i can be any element.
                # But if we assign left-to-right, only APs where i is the max index are fully determined?
                # No, i could be a, a+d, a+2d.
                # So we must check all APs containing i where all OTHER positions are < i.
                
                for ap_idx in pos_to_aps[i]:
                    ap = aps_arr[ap_idx]
                    # Check if all other positions are < i
                    others = [p for p in ap if p != i]
                    if all(o < i for o in others):
                        # Check if adding color c to these others creates < 3 distinct
                        cols = [colors[o] for o in others] + [c]
                        if len(set(cols)) < 3:
                            valid = False
                            break
                if valid:
                    colors[i] = c
                    break
            else:
                colors[i] = 0 # Fallback
                
        # Verify
        if count_violations(np.array(colors), 128) == 0:
            best_solution = colors
            best_count = len(set(colors))
            
    if best_solution is None:
        # Emergency: distinct colors
        best_solution = list(range(128))
        best_count = 128
        
    return best_solution
