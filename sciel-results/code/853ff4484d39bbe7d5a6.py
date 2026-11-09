import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 128
    aps = []
    idx_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for pos, idx in enumerate(ap):
                idx_to_aps[idx].append((ap_idx, pos))
                
    num_aps = len(aps)
    
    def is_valid_ap_colors(c0, c1, c2, c3):
        return len({c0, c1, c2, c3}) >= 3

    # Try k colors
    for k in range(3, 10):
        if time.time() - start_time > time_limit:
            break
            
        restarts = 100 if k == 3 else 20
        for _ in range(restarts):
            if time.time() - start_time > time_limit:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Initial violation count
            violations = [0] * num_aps
            total_violations = 0
            for i, ap in enumerate(aps):
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if not is_valid_ap_colors(c0, c1, c2, c3):
                    violations[i] = 1
                    total_violations += 1
            
            if total_violations == 0:
                return colors
                
            # Local Search
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            steps = 0
            max_steps = 2000
            
            while total_violations > 0 and steps < max_steps and T > min_T:
                if time.time() - start_time > time_limit:
                    break
                    
                idx = random.randint(0, n - 1)
                old_c = colors[idx]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    steps += 1
                    T *= cooling
                    continue
                    
                delta = 0
                for ap_idx, pos in idx_to_aps[idx]:
                    ap = aps[ap_idx]
                    c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    was_valid = (violations[ap_idx] == 0)
                    
                    c[pos] = new_c
                    is_valid_now = is_valid_ap_colors(c[0], c[1], c[2], c[3])
                    
                    if was_valid and not is_valid_now:
                        delta += 1
                    elif not was_valid and is_valid_now:
                        delta -= 1
                        
                if delta < 0 or random.random() < math.exp(-delta / T):
                    colors[idx] = new_c
                    total_violations += delta
                    for ap_idx, pos in idx_to_aps[idx]:
                        ap = aps[ap_idx]
                        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                        if not is_valid_ap_colors(c0, c1, c2, c3):
                            violations[ap_idx] = 1
                        else:
                            violations[ap_idx] = 0
                # else revert
                
                steps += 1
                T *= cooling
                
            if total_violations == 0:
                return colors
                
    # Fallback: Greedy
    colors = []
    for x in range(1, n + 1):
        existing = list(set(colors))
        random.shuffle(existing)
        found = False
        for c in existing:
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1: continue
                c1 = colors[a - 1]
                c2 = colors[a - 1 + d]
                c3 = colors[a - 1 + 2 * d]
                if len({c, c1, c2, c3}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            used_set = set(colors)
            new_c = 0
            while new_c in used_set:
                new_c += 1
            colors.append(new_c)
            
    return colors
