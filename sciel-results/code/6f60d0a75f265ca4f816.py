import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin
    
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
    
    # Precompute color counts for each AP to check validity quickly
    # valid_ap[ap_idx] is True if the AP has >= 3 distinct colors
    
    def get_ap_color_counts(ap, colors):
        return [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
        
    def is_ap_valid(ap, colors):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4}) >= 3

    def count_violations(colors):
        count = 0
        for ap in aps:
            if not is_ap_valid(ap, colors):
                count += 1
        return count

    def is_valid(colors):
        return all(is_ap_valid(ap, colors) for ap in aps)

    # Run the seed greedy
    def run_seed():
        colors = []
        for x in range(1, N + 1):
            found = False
            for color in range(128):
                # Check 4-term AP constraints
                valid_4 = True
                for d in range(1, (x - 1) // 3 + 1):
                    # Positions: x-1, x-1-d, x-1-2d, x-1-3d
                    if x - 1 - 3 * d < 0:
                        continue
                    prev_colors = {colors[x - 1 - j * d] for j in (1, 2, 3)}
                    if len({color} | prev_colors) < 3:
                        valid_4 = False
                        break
                if not valid_4:
                    continue
                
                # Check 3-term AP constraints (necessary condition)
                valid_3 = True
                for d in range(1, (x - 1) // 2 + 1):
                    if x - 1 - 2 * d < 0:
                        continue
                    # For 3-term AP (x-1-2d, x-1-d, x-1), we need not all same
                    # But actually the constraint is only on 4-term APs
                    # The second condition in seed seems to be a heuristic
                    c1 = colors[x - 1 - 2 * d]
                    c2 = colors[x - 1 - d]
                    if c1 == color and c2 == color:
                        valid_3 = False
                        break
                if valid_3:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Fallback: assign a random color
                colors.append(random.randint(0, 127))
        return colors
    
    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    # Local search with incremental violation tracking
    def local_search(colors, target_k, max_time):
        # Initialize violation counts
        ap_valid = [is_ap_valid(ap, colors) for ap in aps]
        total_violations = sum(1 for v in ap_valid if not v)
        
        # Precompute for each position, the list of APs it belongs to
        # We already have ap_indices
        
        while time.time() < max_time:
            if total_violations == 0:
                return colors, 0
                
            # Pick a random violated AP
            violated_aps = [i for i, v in enumerate(ap_valid) if not v]
            if not violated_aps:
                return colors, total_violations
            ap_idx = random.choice(violated_aps)
            ap = aps[ap_idx]
            
            best_new_violations = total_violations
            best_idx = -1
            best_new_c = -1
            
            # Try changing each position in the AP to each possible color
            for pos in range(4):
                idx = ap[pos]
                old_c = colors[idx]
                for new_c in range(target_k):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta in violations
                    delta = 0
                    # For each AP containing idx, check if validity changes
                    for a_idx in ap_indices[idx]:
                        a = aps[a_idx]
                        # Get colors of a
                        # We need to simulate the change
                        # old colors: colors[a[0]], ..., colors[a[3]]
                        # new colors: replace colors[idx] with new_c
                        
                        # Efficient check:
                        # Count distinct colors in a before and after
                        # Since only one position changes, we can do this quickly
                        
                        # Let's just recompute for the 4 elements of a
                        # This is O(1) per AP
                        c0 = colors[a[0]] if a[0] != idx else new_c
                        c1 = colors[a[1]] if a[1] != idx else new_c
                        c2 = colors[a[2]] if a[2] != idx else new_c
                        c3 = colors[a[3]] if a[3] != idx else new_c
                        
                        new_valid = len({c0, c1, c2, c3}) >= 3
                        
                        if ap_valid[a_idx] and not new_valid:
                            delta += 1
                        elif not ap_valid[a_idx] and new_valid:
                            delta -= 1
                    
                    new_violations = total_violations + delta
                    if new_violations < best_new_violations:
                        best_new_violations = new_violations
                        best_idx = idx
                        best_new_c = new_c
            
            if best_idx >= 0:
                # Apply the best move
                old_c = colors[best_idx]
                colors[best_idx] = best_new_c
                
                # Update ap_valid and total_violations
                for a_idx in ap_indices[best_idx]:
                    a = aps[a_idx]
                    c0 = colors[a[0]]
                    c1 = colors[a[1]]
                    c2 = colors[a[2]]
                    c3 = colors[a[3]]
                    new_valid = len({c0, c1, c2, c3}) >= 3
                    if ap_valid[a_idx] != new_valid:
                        if new_valid:
                            total_violations -= 1
                        else:
                            total_violations += 1
                        ap_valid[a_idx] = new_valid
            else:
                # No improving move found, try a random move to escape local minimum
                pos = random.randint(0, 3)
                idx = ap[pos]
                old_c = colors[idx]
                new_c = random.randint(0, target_k - 1)
                if new_c != old_c:
                    colors[idx] = new_c
                    for a_idx in ap_indices[idx]:
                        a = aps[a_idx]
                        c0 = colors[a[0]]
                        c1 = colors[a[1]]
                        c2 = colors[a[2]]
                        c3 = colors[a[3]]
                        new_valid = len({c0, c1, c2, c3}) >= 3
                        if ap_valid[a_idx] != new_valid:
                            if new_valid:
                                total_violations -= 1
                            else:
                                total_violations += 1
                            ap_valid[a_idx] = new_valid
                    if total_violations > best_new_violations:
                         # If it got worse, we could revert, but random walk is okay for diversity
                         # Let's just keep it and see
                         pass
                         
        return colors, total_violations

    # Try to improve with local search
    for target_k in range(3, best_k + 1):
        if time.time() > DEADLINE - 2:
            break
            
        k_start = time.time()
        k_budget = 2.0
        
        for restart in range(100):
            if time.time() > k_start + k_budget:
                break
                
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            colors, violations = local_search(colors, target_k, k_start + k_budget)
            
            if violations == 0:
                best_colors = colors[:]
                best_k = target_k
                break
                
    # Final validation
    if not is_valid(best_colors):
        # Fallback to seed if it's valid, else return seed
        if is_valid(seed_colors):
            best_colors = seed_colors
    
    return best_colors
