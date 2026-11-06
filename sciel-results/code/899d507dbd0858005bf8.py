import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(n):
        for d in range(1, n):
            if a + 3 * d < n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def is_valid(cols):
        for ap in aps:
            if len(set(cols[i] for i in ap)) < 3:
                return False
        return True
    
    def num_colors(cols):
        return len(set(cols))
    
    # Initial valid coloring
    colors = [0] * n
    for x in range(n):
        for c in range(128):
            ok = True
            for ap_idx in pos_to_aps[x]:
                ap = aps[ap_idx]
                if ap[3] != x:
                    continue
                if len({c, colors[ap[0]], colors[ap[1]], colors[ap[2]]}) < 3:
                    ok = False
                    break
            if ok:
                colors[x] = c
                break
    
    best = colors[:]
    best_n = num_colors(best)
    
    random.seed(42)
    
    # Local search: try to reduce colors
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random position
        pos = random.randint(0, n - 1)
        old_c = best[pos]
        
        # Try a new color
        new_c = random.randint(0, 127)
        if new_c == old_c:
            continue
        
        best[pos] = new_c
        
        # Check if still valid
        valid = True
        for ap_idx in pos_to_aps[pos]:
            ap = aps[ap_idx]
            if len(set(best[i] for i in ap)) < 3:
                valid = False
                break
        
        if valid:
            new_n = num_colors(best)
            if new_n < best_n:
                best_n = new_n
        else:
            best[pos] = old_c
    
    return best
