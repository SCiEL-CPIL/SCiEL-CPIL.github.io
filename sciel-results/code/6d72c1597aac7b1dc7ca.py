import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    aps = []
    for d in range(1, n):
        for a in range(0, n - 3*d):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def check_position(pos, c, colors):
        for ap_idx in pos_aps[pos]:
            ap = aps[ap_idx]
            if all(colors[p] != -1 for p in ap):
                if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                    return False
        return True
    
    def solve(num_colors, seed, max_depth):
        random.seed(seed)
        colors = [-1] * n
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == max_depth:
                for p in range(max_depth, n):
                    found = False
                    for c in range(num_colors):
                        if check_position(p, c, colors):
                            colors[p] = c
                            found = True
                            break
                    if not found:
                        return False
                return True
            
            color_order = list(range(num_colors))
            random.shuffle(color_order)
            
            for c in color_order:
                colors[pos] = c
                if check_position(pos, c, colors):
                    if backtrack(pos + 1):
                        return True
                colors[pos] = -1
            return False
        
        return backtrack(0)
    
    for depth in [12, 15, 18, 20]:
        for seed in range(50):
            if time.time() > deadline - 30:
                break
            if solve(3, seed, depth):
                return colors
    
    for depth in [12, 15, 18]:
        for seed in range(40):
            if time.time() > deadline - 10:
                break
            if solve(4, seed, depth):
                return colors
    
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x-1)//3 + 1):
                if len({color} | {colors[x-1-j*d] for j in (1, 2, 3)}) < 3:
                    ok = False
                    break
            if not ok:
                continue
            for d in range(1, (x-1)//2 + 1):
                if not any(colors[x-1-j*d] != color for j in (1, 2)):
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
