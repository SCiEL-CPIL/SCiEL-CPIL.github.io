"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_aps(n):
    aps = []
    pos_aps = [[] for _ in range(n)]
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            idx = len(aps)
            t = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            aps.append(t)
            for p in t:
                pos_aps[p].append(idx)
    return aps, pos_aps

def construct():
    t0 = time.time()
    deadline = t0 + 72.0
    n = 128
    aps, pos_aps = _build_aps(n)
    
    best = None
    best_k = 128
    
    for nc in [3, 4, 5, 6, 8]:
        if time.time() > deadline - 5:
            break
        for restart in range(400):
            if time.time() > deadline - 2:
                break
            
            if restart == 0:
                colors = np.array([i % nc for i in range(n)], dtype=np.int8)
            elif restart == 1:
                colors = np.array([(i * 7) % nc for i in range(n)], dtype=np.int8)
            elif restart == 2:
                colors = np.array([(i * 13) % nc for i in range(n)], dtype=np.int8)
            elif restart == 3:
                colors = np.array([(i * 5 + i * i) % nc for i in range(n)], dtype=np.int8)
            elif restart == 4:
                colors = np.array([(i * 11 + 2 * i * i) % nc for i in range(n)], dtype=np.int8)
            else:
                colors = np.random.randint(0, nc, size=n, dtype=np.int8)
            
            v = 0
            for a, b, c, d in aps:
                if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                    v += 1
            
            if v == 0:
                k = len(set(colors.tolist()))
                if k < best_k:
                    best = colors.copy()
                    best_k = k
                break
            
            for it in range(15000):
                if time.time() > deadline - 2:
                    break
                if v == 0:
                    break
                
                pos = random.randint(0, n - 1)
                cur = colors[pos]
                
                best_delta = 0
                best_c = cur
                for new_c in range(nc):
                    if new_c == cur:
                        continue
                    delta = 0
                    for ai in pos_aps[pos]:
                        a, b, c, d = aps[ai]
                        old_s = {colors[a], colors[b], colors[c], colors[d]}
                        old_bad = 1 if len(old_s) < 3 else 0
                        vals = [colors[a], colors[b], colors[c], colors[d]]
                        if a == pos:
                            vals[0] = new_c
                        elif b == pos:
                            vals[1] = new_c
                        elif c == pos:
                            vals[2] = new_c
                        elif d == pos:
                            vals[3] = new_c
                        new_bad = 1 if len(set(vals)) < 3 else 0
                        delta += new_bad - old_bad
                    if delta < best_delta or (delta == best_delta and random.random() < 0.15):
                        best_delta = delta
                        best_c = new_c
                
                if best_c != cur:
                    colors[pos] = best_c
                    v += best_delta
            
            if v == 0:
                k = len(set(colors.tolist()))
                if k < best_k:
                    best = colors.copy()
                    best_k = k
                break
    
    if best is not None:
        return best.tolist()
    
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d < 1:
                    break
                s = {color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]}
                if len(s) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
