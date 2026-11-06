import numpy as np
import time
import random

def construct():
    N = 128
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to AP indices
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def is_bad(c0, c1, c2, c3):
        return len({c0, c1, c2, c3}) < 3

    def calculate_cost(colors):
        cost = 0
        for ap in aps:
            if is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                cost += 1
        return cost

    def get_delta(pos, old_c, new_c, colors):
        delta = 0
        for idx in pos_to_aps[pos]:
            ap = aps[idx]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            old_bad = 1 if is_bad(c0, c1, c2, c3) else 0
            
            if ap[0] == pos:
                c0 = new_c
            elif ap[1] == pos:
                c1 = new_c
            elif ap[2] == pos:
                c2 = new_c
            elif ap[3] == pos:
                c3 = new_c
            
            new_bad = 1 if is_bad(c0, c1, c2, c3) else 0
            
            if old_bad == 0 and new_bad == 1:
                delta += 1
            elif old_bad == 1 and new_bad == 0:
                delta -= 1
        return delta

    start_time = time.time()
    deadline = start_time + 85.0

    best_valid = None
    best_valid_colors = None
    
    # Try modular constructions first
    for m in range(3, 20):
        if time.time() > deadline:
            break
        colors = np.array([i % m for i in range(N)], dtype=np.int64)
        cost = calculate_cost(colors)
        if cost == 0:
            return colors.tolist()
    
    # Try K from 5 upwards with local search
    for K in range(5, 15):
        if time.time() > deadline:
            break
            
        restarts = max(5, 30 - K)
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            colors = np.random.randint(0, K, N)
            current_cost = calculate_cost(colors)
            
            if current_cost == 0:
                return colors.tolist()
                
            T = 5.0
            T_min = 0.1
            cooling = 0.995
            max_steps = 3000
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                pos = np.random.randint(N)
                old_c = colors[pos]
                new_c = np.random.randint(K)
                if new_c == old_c:
                    continue
                    
                delta = get_delta(pos, old_c, new_c, colors)
                
                if delta <= 0:
                    colors[pos] = new_c
                    current_cost += delta
                else:
                    if np.random.rand() < np.exp(-delta / T):
                        colors[pos] = new_c
                        current_cost += delta
                
                if current_cost == 0:
                    return colors.tolist()
                    
                T *= cooling
                if T < T_min:
                    break

    # Fallback: Try larger K with quick search
    for K in [20, 50, 100]:
        if time.time() > deadline:
            break
        for r in range(5):
            if time.time() > deadline:
                break
            colors = np.random.randint(0, K, N)
            cost = calculate_cost(colors)
            if cost == 0:
                return colors.tolist()
            T = 2.0
            for step in range(500):
                if time.time() > deadline:
                    break
                pos = np.random.randint(N)
                old_c = colors[pos]
                new_c = np.random.randint(K)
                if new_c == old_c: continue
                delta = get_delta(pos, old_c, new_c, colors)
                if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                    colors[pos] = new_c
                    cost += delta
                if cost == 0:
                    return colors.tolist()
                T *= 0.98

    # Last resort: Random with K=128, verify
    colors = np.random.randint(0, 128, N)
    if calculate_cost(colors) == 0:
        return colors.tolist()
    
    return colors.tolist()
