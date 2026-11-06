import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 128
    
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
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    def check_ap_valid(ap, colors):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4}) >= 3
    
    def run_seed():
        colors = []
        for x in range(1, N + 1):
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    start = x - 1 - 3 * d
                    if start < 0:
                        break
                    c1 = colors[start]
                    c2 = colors[start + d]
                    c3 = colors[start + 2 * d]
                    if len({color, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
        return colors
    
    seed_colors = run_seed()
    best_colors = seed_colors
    best_k = count_distinct_colors(seed_colors)
    
    for target_k in range(3, best_k):
        if time.time() > DEADLINE - 3:
            break
            
        k_start = time.time()
        k_budget = 5.0
        
        for restart in range(5000):
            if time.time() > k_start + k_budget:
                break
            
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            violation_flags = [False] * num_aps
            num_violations = 0
            for ai, ap in enumerate(aps):
                if not check_ap_valid(ap, colors):
                    violation_flags[ai] = True
                    num_violations += 1
            
            max_steps = 20000
            for step in range(max_steps):
                if time.time() > k_start + k_budget:
                    break
                if num_violations == 0:
                    break
                    
                violated = [i for i, v in enumerate(violation_flags) if v]
                if not violated:
                    break
                    
                ai = random.choice(violated)
                ap = aps[ai]
                
                moved = False
                for pos in range(4):
                    idx = ap[pos]
                    old_c = colors[idx]
                    # Save old flags for affected APs
                    affected = ap_indices[idx]
                    old_flags = {aff_ai: violation_flags[aff_ai] for aff_ai in affected}
                    
                    for new_c in range(target_k):
                        if new_c == old_c:
                            continue
                        colors[idx] = new_c
                        
                        old_nv = num_violations
                        for aff_ai in affected:
                            was_v = old_flags[aff_ai]
                            is_v = not check_ap_valid(aps[aff_ai], colors)
                            if was_v and not is_v:
                                num_violations -= 1
                            elif not was_v and is_v:
                                num_violations += 1
                            violation_flags[aff_ai] = is_v
                        
                        if num_violations < old_nv:
                            moved = True
                            break
                        else:
                            # Revert
                            colors[idx] = old_c
                            for aff_ai in affected:
                                violation_flags[aff_ai] = old_flags[aff_ai]
                            num_violations = old_nv
                    if moved:
                        break
                
                if not moved:
                    idx = random.randint(0, N - 1)
                    old_c = colors[idx]
                    affected = ap_indices[idx]
                    old_flags = {aff_ai: violation_flags[aff_ai] for aff_ai in affected}
                    new_c = random.randint(0, target_k - 1)
                    if new_c == old_c:
                        continue
                    colors[idx] = new_c
                    old_nv = num_violations
                    for aff_ai in affected:
                        was_v = old_flags[aff_ai]
                        is_v = not check_ap_valid(aps[aff_ai], colors)
                        if was_v and not is_v:
                            num_violations -= 1
                        elif not was_v and is_v:
                            num_violations += 1
                        violation_flags[aff_ai] = is_v
                    if num_violations > old_nv:
                        colors[idx] = old_c
                        for aff_ai in affected:
                            violation_flags[aff_ai] = old_flags[aff_ai]
                        num_violations = old_nv
            
            if num_violations == 0:
                k_used = count_distinct_colors(colors)
                if k_used < best_k:
                    best_colors = colors
                    best_k = k_used
                break
    
    return best_colors
