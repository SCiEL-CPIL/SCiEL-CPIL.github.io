import time
import random

def construct():
    deadline = time.time() + 58
    N = 128
    
    # Precompute APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(aps)
    
    # Map index to list of AP indices it belongs to
    index_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(i)

    def solve(k, time_limit):
        colors = [random.randint(0, k - 1) for _ in range(N)]
        
        is_bad = [False] * num_aps
        bad_count = 0
        for i, ap in enumerate(aps):
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                is_bad[i] = True
                bad_count += 1
                
        if bad_count == 0:
            return colors

        best_bad_count = bad_count
        best_colors = colors[:]
        
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        start_time = time.time()
        
        while temp > min_temp:
            if time.time() > start_time + time_limit:
                break
                
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            
            # Pick a new color randomly
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue

            # Calculate delta
            delta = 0
            for ap_idx in index_to_aps[pos]:
                ap = aps[ap_idx]
                was_bad = is_bad[ap_idx]
                
                c0 = colors[ap[0]] if ap[0] != pos else new_color
                c1 = colors[ap[1]] if ap[1] != pos else new_color
                c2 = colors[ap[2]] if ap[2] != pos else new_color
                c3 = colors[ap[3]] if ap[3] != pos else new_color
                is_now_bad = len({c0, c1, c2, c3}) < 3
                
                if was_bad and not is_now_bad:
                    delta -= 1
                elif not was_bad and is_now_bad:
                    delta += 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < pow(2.71828, -delta / temp):
                    accept = True
            
            if accept:
                colors[pos] = new_color
                for ap_idx in index_to_aps[pos]:
                    ap = aps[ap_idx]
                    c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    is_now_bad = len(c) < 3
                    was_bad = is_bad[ap_idx]
                    
                    if was_bad and not is_now_bad:
                        bad_count -= 1
                    elif not was_bad and is_now_bad:
                        bad_count += 1
                    
                    is_bad[ap_idx] = is_now_bad
                
                if bad_count < best_bad_count:
                    best_bad_count = bad_count
                    best_colors = colors[:]
                    if bad_count == 0:
                        return colors
            else:
                # Revert (do nothing to colors, is_bad, bad_count)
                pass

            temp *= cooling
            
        return best_colors if best_bad_count == 0 else None

    # Try 3 colors
    res = solve(3, 50)
    if res:
        return res

    # Try 4 colors
    res = solve(4, 10)
    if res:
        return res

    # Fallback
    return [random.randint(0, 2) for _ in range(N)]
