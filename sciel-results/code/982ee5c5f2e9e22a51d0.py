import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Precompute all nonconstant 4-term APs: (a, a+d, a+2d, a+3d)
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            if a + 3 * d <= n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
            else:
                break
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # For each AP, store positions as 0-indexed
    aps0 = [(a-1, b-1, c-1, d-1) for (a, b, c, d) in aps]
    pos_to_aps0 = [[] for _ in range(n)]
    for idx, ap in enumerate(aps0):
        for pos in ap:
            pos_to_aps0[pos].append(idx)
    
    def count_violations(colors):
        """Count APs with < 3 distinct colors."""
        cnt = 0
        for ap in aps0:
            if len(set(colors[p] for p in ap)) < 3:
                cnt += 1
        return cnt
    
    def simulated_annealing(k, max_iters=200000, seed=0):
        rng = random.Random(seed)
        
        # Random initialization
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        
        # Compute initial violations
        violated = [False] * num_aps
        for idx in range(num_aps):
            ap = aps0[idx]
            violated[idx] = len(set(colors[p] for p in ap)) < 3
        num_violations = sum(violated)
        
        if num_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = num_violations
        
        # Simulated annealing schedule
        T_start = 1.0
        T_end = 0.001
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            if num_violations == 0:
                return colors
            
            # Pick a random position to recolor
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            
            if k > 1:
                new_color = rng.randint(0, k - 1)
                if new_color == old_color:
                    new_color = (new_color + 1) % k
            
            colors[pos] = new_color
            
            # Compute delta violations
            delta = 0
            for ap_i in pos_to_aps0[pos]:
                was_v = violated[ap_i]
                vals = set(colors[q] for q in aps0[ap_i])
                is_v = len(vals) < 3
                violated[ap_i] = is_v
                if is_v and not was_v:
                    delta += 1
                elif not is_v and was_v:
                    delta -= 1
            
            new_violations = num_violations + delta
            
            # Accept or reject
            if new_violations < num_violations:
                num_violations = new_violations
                if num_violations < best_violations:
                    best_violations = num_violations
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta / T)
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    num_violations = new_violations
                else:
                    # Revert
                    colors[pos] = old_color
                    for ap_i in pos_to_aps0[pos]:
                        vals = set(colors[q] for q in aps0[ap_i])
                        violated[ap_i] = len(vals) < 3
                    num_violations = sum(violated)
            
            T *= cooling
        
        return best_colors if best_violations == 0 else None
    
    # Try k=3 with simulated annealing
    for seed in range(50):
        if time.time() > deadline:
            break
        result = simulated_annealing(3, max_iters=100000, seed=seed)
        if result is not None:
            return result
    
    # If 3 colors fail, try k=4
    for seed in range(30):
        if time.time() > deadline:
            break
        result = simulated_annealing(4, max_iters=100000, seed=seed)
        if result is not None:
            return result
    
    # Fallback: deterministic greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            for d in range(1, (x - 1) // 2 + 1):
                if not any(colors[x - 1 - j * d] != color for j in (1, 2)):
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
    return colors
