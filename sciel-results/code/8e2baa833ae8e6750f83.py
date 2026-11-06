import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs
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
    
    def count_colors_in_ap(ap, colors):
        """Return number of distinct colors in the AP."""
        c = colors[ap[0]]
        count = 1
        for j in range(1, 4):
            cj = colors[ap[j]]
            if cj != c:
                # Check if cj matches any other
                found = False
                for k in range(j):
                    if colors[ap[k]] == cj:
                        found = True
                        break
                if not found:
                    count += 1
        return count
    
    def is_ap_valid(ap, colors):
        """AP is valid if it uses at least 3 distinct colors."""
        return count_colors_in_ap(ap, colors) >= 3
    
    def run_seed():
        """Greedy seed coloring."""
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    # Check AP ending at x: (x-3d, x-2d, x-d, x)
                    # and APs where x is not the last element
                    pass
                # Simpler: check all APs containing position x-1
                pos = x - 1
                for ai in ap_indices[pos]:
                    ap = aps[ai]
                    # Temporarily assign color to pos
                    test_colors = colors + [color]
                    if len(test_colors) < 4:
                        continue
                    # Check if this AP is valid with the new color
                    c0 = test_colors[ap[0]]
                    c1 = test_colors[ap[1]]
                    c2 = test_colors[ap[2]]
                    c3 = test_colors[ap[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                colors.append(0)  # Fallback
        return colors
    
    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    # Better seed: try to use fewer colors
    # Try k=3,4,5,... and see if we can find a valid coloring
    
    def try_local_search(target_k, max_steps, seed=None):
        """Local search to find valid coloring with target_k colors."""
        if seed is None:
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
        else:
            colors = seed[:]
        
        # Initialize violation state
        violation_flags = [False] * num_aps
        num_violations = 0
        for ai, ap in enumerate(aps):
            if not is_ap_valid(ap, colors):
                violation_flags[ai] = True
                num_violations += 1
        
        if num_violations == 0:
            return colors, 0
        
        step = 0
        stuck_count = 0
        best_local_violations = num_violations
        
        while num_violations > 0 and step < max_steps:
            step += 1
            if time.time() > DEADLINE:
                break
            
            # Pick a random violated AP
            violated_list = [i for i, v in enumerate(violation_flags) if v]
            if not violated_list:
                break
            ai = random.choice(violated_list)
            ap = aps[ai]
            
            # Pick a random element in the AP to change
            pos_in_ap = random.randint(0, 3)
            idx = ap[pos_in_ap]
            old_c = colors[idx]
            
            # Try all possible new colors for this position
            best_new_c = old_c
            best_delta = 0  # We want to minimize violations
            
            for new_c in range(target_k):
                if new_c == old_c:
                    continue
                
                # Calculate what happens if we change colors[idx] to new_c
                # Only need to check APs containing idx
                old_flags_snapshot = {}
                delta = 0
                for affected_ai in ap_indices[idx]:
                    old_violated = violation_flags[affected_ai]
                    # Check if this AP would be violated with new color
                    ap2 = aps[affected_ai]
                    # Build the color tuple
                    c_vals = []
                    for p in ap2:
                        if p == idx:
                            c_vals.append(new_c)
                        else:
                            c_vals.append(colors[p])
                    is_violated = len(set(c_vals)) < 3
                    
                    if old_violated and not is_violated:
                        delta -= 1
                    elif not old_violated and is_violated:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_new_c == old_c:
                stuck_count += 1
                if stuck_count > 50:
                    # Random perturbation
                    idx = random.randint(0, N - 1)
                    old_c = colors[idx]
                    new_c = random.randint(0, target_k - 1)
                    if new_c == old_c:
                        continue
                    colors[idx] = new_c
                    # Recompute all violations
                    num_violations = 0
                    for ai2 in range(num_aps):
                        if not is_ap_valid(aps[ai2], colors):
                            violation_flags[ai2] = True
                            num_violations += 1
                        else:
                            violation_flags[ai2] = False
                    stuck_count = 0
                    continue
                continue
            
            # Apply the best move
            colors[idx] = best_new_c
            
            # Update violation flags
            for affected_ai in ap_indices[idx]:
                was_violated = violation_flags[affected_ai]
                is_violated = not is_ap_valid(aps[affected_ai], colors)
                
                if was_violated and not is_violated:
                    num_violations -= 1
                elif not was_violated and is_violated:
                    num_violations += 1
                violation_flags[affected_ai] = is_violated
            
            if num_violations < best_local_violations:
                best_local_violations = num_violations
                stuck_count = 0
        
        return colors, num_violations
    
    # Try to find colorings with fewer colors
    for target_k in range(3, best_k):
        if time.time() > DEADLINE - 3:
            break
        
        k_start = time.time()
        k_budget = min(4.0, DEADLINE - time.time() - 3)
        
        found_valid = False
        for restart in range(200):
            if time.time() > k_start + k_budget:
                break
            
            colors, num_violations = try_local_search(target_k, max_steps=2000)
            
            if num_violations == 0:
                k_used = len(set(colors))
                if k_used < best_k:
                    best_colors = colors
                    best_k = k_used
                    found_valid = True
                    break
        
        if found_valid:
            break
    
    # Also try to improve the seed with local search using its own color count
    if time.time() < DEADLINE - 3:
        seed_k = len(set(seed_colors))
        for restart in range(50):
            if time.time() > DEADLINE - 3:
                break
            colors, num_violations = try_local_search(seed_k, max_steps=3000, seed=seed_colors)
            if num_violations == 0:
                k_used = len(set(colors))
                if k_used < best_k:
                    best_colors = colors
                    best_k = k_used
                break
    
    return best_colors
