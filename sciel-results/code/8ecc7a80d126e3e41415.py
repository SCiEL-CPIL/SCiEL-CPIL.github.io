"""Colorings with three colors in every four-term progression."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precompute all 4-term APs
def _precompute_aps():
    aps = []
    for start in range(1, 129):
        max_d = (128 - start) // 3
        for d in range(1, max_d + 1):
            a, b, c, e = start, start + d, start + 2*d, start + 3*d
            if e <= 128:
                aps.append((a-1, b-1, c-1, e-1))
    return aps

_ALL_APS = _precompute_aps()
_N = len(_ALL_APS)

# Precompute index arrays for vectorized operations
_A_IDX = np.array([ap[0] for ap in _ALL_APS], dtype=np.int32)
_B_IDX = np.array([ap[1] for ap in _ALL_APS], dtype=np.int32)
_C_IDX = np.array([ap[2] for ap in _ALL_APS], dtype=np.int32)
_E_IDX = np.array([ap[3] for ap in _ALL_APS], dtype=np.int32)

def _count_violations(colors):
    """Count APs with fewer than 3 distinct colors (i.e., 1 or 2 distinct)."""
    ca = colors[_A_IDX]
    cb = colors[_B_IDX]
    cc = colors[_C_IDX]
    ce = colors[_E_IDX]
    
    # 1 distinct: all equal
    all_eq = (ca == cb) & (cb == cc) & (cc == ce)
    
    # 2 distinct: 
    # (i) three equal, one different
    t1 = (ca == cb) & (cb == cc) & (cc != ce)
    t2 = (ca == cb) & (cb == ce) & (ce != cc)
    t3 = (ca == cc) & (cc == ce) & (ce != cb)
    t4 = (cb == cc) & (cc == ce) & (ce != ca)
    # (ii) two pairs
    p1 = (ca == cb) & (cc == ce) & (ca != cc)
    p2 = (ca == cc) & (cb == ce) & (ca != cb)
    p3 = (ca == ce) & (cb == cc) & (ca != cb)
    
    two_distinct = t1 | t2 | t3 | t4 | p1 | p2 | p3
    
    return int(np.sum(all_eq | two_distinct))

def construct():
    t0 = time.time()
    deadline = t0 + 80
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Start with random 3-coloring
    colors = np.random.randint(0, 3, size=128, dtype=np.int8)
    v = _count_violations(colors)
    best_colors = colors.copy()
    best_v = v
    
    # Simulated annealing
    T = 10.0
    T_min = 0.001
    cooling = 0.999
    
    while T > T_min and time.time() < deadline:
        pos = random.randint(0, 127)
        old = colors[pos]
        new = random.randint(0, 2)
        if new == old:
            continue
        
        colors[pos] = new
        nv = _count_violations(colors)
        d = nv - v
        if d <= 0 or random.random() < np.exp(-d / T):
            v = nv
            if v < best_v:
                best_v = v
                best_colors = colors.copy()
        else:
            colors[pos] = old
        
        T *= cooling
    
    colors = best_colors.copy()
    
    # Local search polish
    if best_v > 0:
        for _ in range(100000):
            if time.time() > deadline or best_v == 0:
                break
            pos = random.randint(0, 127)
            old = colors[pos]
            new = random.randint(0, 2)
            if new == old:
                continue
            colors[pos] = new
            nv = _count_violations(colors)
            if nv <= best_v:
                best_v = nv
                best_colors = colors.copy()
                colors = best_colors.copy()
            else:
                colors[pos] = old
    
    colors = best_colors
    v = _count_violations(colors)
    
    if v == 0:
        return [int(c) for c in colors]
    
    # Fallback to greedy
    colors_list = []
    for x in range(1, 129):
        placed = False
        for color in range(3):
            ok = True
            for d in range(1, (x-1)//3 + 1):
                idx = [x-1-j*d for j in range(4)]
                if any(i < 0 for i in idx):
                    continue
                vals = [color if i == x-1 else colors_list[i] for i in idx]
                if len(set(vals)) < 3:
                    ok = False
                    break
            if ok:
                colors_list.append(color)
                placed = True
                break
        if not placed:
            # Should not happen with 3 colors, but fallback
            for color in range(128):
                ok = True
                for d in range(1, (x-1)//3 + 1):
                    idx = [x-1-j*d for j in range(4)]
                    if any(i < 0 for i in idx):
                        continue
                    vals = [color if i == x-1 else colors_list[i] for i in idx]
                    if len(set(vals)) < 3:
                        ok = False
                        break
                if ok:
                    colors_list.append(color)
                    placed = True
                    break
            if not placed:
                colors_list.append(0)
    return colors_list

# EVOLVE-BLOCK-END
