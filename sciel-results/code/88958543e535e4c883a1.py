import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 128
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a - 1, b - 1, c - 1, e - 1))
                
    # Greedy baseline
    def greedy():
        colors = [-1] * N
        for i in range(N):
            for col in range(128):
                valid = True
                for d in range(1, (i // 3) + 1):
                    idx3 = i - 3 * d
                    if idx3 < 0: break
                    idx2 = i - 2 * d
                    idx1 = i - d
                    c3 = colors[idx3]
                    c2 = colors[idx2]
                    c1 = colors[idx1]
                    if c3 == -1 or c2 == -1 or c1 == -1:
                        continue
                    s = {c3, c2, c1, col}
                    if len(s) < 3:
                        valid = False
                        break
                if valid:
                    colors[i] = col
                    break
            else:
                colors[i] = 0
        return colors

    baseline_colors = greedy()
    K_base = len(set(baseline_colors))
    
    if K_base == 3:
        return baseline_colors

    index_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(i)
            
    def count_bad(colors):
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad
        
    def sa_search(K, time_limit):
        colors = np.random.randint(0, K, N)
        current_bad = count_bad(colors)
        
        if current_bad == 0:
            return colors, current_bad
            
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            if T < T_min:
                break
            idx = random.randint(0, N - 1)
            old_color = colors[idx]
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
                
            delta = 0
            for ap_idx in index_to_aps[idx]:
                ap = aps[ap_idx]
                c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                old_bad = 1 if len(set(c)) < 3 else 0
                
                for j, pos in enumerate(ap):
                    if pos == idx:
                        c[j] = new_color
                new_bad = 1 if len(set(c)) < 3 else 0
                
                delta += new_bad - old_bad
                
            new_bad = current_bad + delta
            
            if new_bad < current_bad or random.random() < pow(2.71828, -delta / T):
                colors[idx] = new_color
                current_bad = new_bad
                
                if current_bad == 0:
                    return colors, current_bad
                    
            T *= cooling
            
        return colors, current_bad

    # Try K from 3 to K_base - 1
    for K in range(3, K_base):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Allocate time to this K
        # If K=3, give it more time?
        # Just give it min(15, time_left)
        t_alloc = min(15, time_left)
        
        res_colors, res_bad = sa_search(K, t_alloc)
        
        if res_bad == 0:
            return list(res_colors)
            
    return baseline_colors
