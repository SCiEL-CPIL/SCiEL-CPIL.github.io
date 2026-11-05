import time

def construct():
    # Generate seed coloring
    colors_seed = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                i0 = x - 1
                i1 = x - 1 - d
                i2 = x - 1 - 2*d
                i3 = x - 1 - 3*d
                if i3 < 0: continue
                prev = {colors_seed[i1], colors_seed[i2], colors_seed[i3]}
                if len({color} | prev) < 3:
                    valid = False
                    break
            if valid:
                colors_seed.append(color)
                break
    K_seed = len(set(colors_seed))
    
    # Try to improve
    deadline = time.time() + 80
    # Precompute APs ending at each x
    # aps[x] = list of (i1, i2, i3) for x-1, x-1-d, x-1-2d, x-1-3d
    # Actually we just need to check the set size.
    
    for K in range(3, K_seed):
        if time.time() > deadline:
            break
        start_time = time.time()
        colors = [-1] * 128
        colors[0] = 0
        if backtrack(1, colors, K, start_time + 10): # 10s per K
            return colors
            
    return colors_seed

def backtrack(pos, colors, K, deadline):
    if time.time() > deadline:
        return False
    if pos == 128:
        return True
    
    for c in range(K):
        colors[pos] = c
        valid = True
        x = pos + 1
        for d in range(1, (x-1)//3 + 1):
            i3 = x - 1 - 3*d
            if i3 < 0: continue
            c3 = colors[i3]
            c2 = colors[i3 + d]
            c1 = colors[i3 + 2*d]
            c0 = c
            if len({c0, c1, c2, c3}) < 3:
                valid = False
                break
        
        if valid:
            if backtrack(pos + 1, colors, K, deadline):
                return True
        
        colors[pos] = -1
    return False
