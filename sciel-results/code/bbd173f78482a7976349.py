import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
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
    
    def check_ap_valid(ap, colors):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4}) >= 3
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            if not check_ap_valid(ap, colors):
                count += 1
        return count
    
    def is_valid(colors):
        return all(check_ap_valid(ap, colors) for ap in aps)
    
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
    best_colors = seed_colors
    best_k = len(set(seed_colors))
    
    # Try to improve with local search
    for target_k in range(3, best_k):
        if time.time() > DEADLINE - 3:
            break
            
        k_start = time.time()
        k_budget = 3.0
        
        for restart in range(200):
            if time.time() > k_start + k_budget:
                break
                
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            # Local search with greedy best-move
            for step in range(5000):
                if time.time() > k_start + k_budget:
                    break
                    
                violations = count_violations(colors)
                if violations == 0:
                    # Found valid coloring with target_k colors
                    best_colors = colors[:]
                    best_k = target_k
                    break
                    
                # Pick a random violated AP
                violated_aps = [ap for ap in aps if not check_ap_valid(ap, colors)]
                if not violated_aps:
                    break
                ap = random.choice(violated_aps)
                
                # Try all positions in the AP and all colors, find best move
                best_new_violations = violations
                best_idx = -1
                best_new_c = -1
                
                for pos in range(4):
                    idx = ap[pos]
                    old_c = colors[idx]
                    for new_c in range(target_k):
                        if new_c == old_c:
                            continue
                        colors[idx] = new_c
                        new_violations = count_violations(colors)
                        if new_violations < best_new_violations:
                            best_new_violations = new_violations
                            best_idx = idx
                            best_new_c = new_c
                        colors[idx] = old_c
                
                if best_idx >= 0:
                    colors[best_idx] = best_new_c
                    violations = best_new_violations
                else:
                    # No improving move found, try random move
                    pos = random.randint(0, 3)
                    idx = ap[pos]
                    old_c = colors[idx]
                    new_c = random.randint(0, target_k - 1)
                    if new_c != old_c:
                        colors[idx] = new_c
                        new_violations = count_violations(colors)
                        if new_violations <= violations:
                            violations = new_violations
                        else:
                            colors[idx] = old_c
                    
            if best_k == target_k:
                break
    
    # Final validation
    if not is_valid(best_colors):
        # Fallback to seed if it's valid, else return seed
        if is_valid(seed_colors):
            best_colors = seed_colors
    
    return best_colors
