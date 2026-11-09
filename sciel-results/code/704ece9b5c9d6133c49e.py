import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total_points = q * q
    k_const = 3
    
    idx_to_elem = [(i // p, i % p) for i in range(q)]
    
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb

    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    directions = [(0, 1)] + [(1, m) for m in range(q)]
    num_dirs = len(directions)
    
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(1, q, dtype=np.int32)
    
    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for v_idx, (dx, dy) in enumerate(directions):
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        line_other[:, v_idx, :] = nx * q + ny

    deadline = time.time() + 570.0
    best_size = 0
    best_E = np.zeros(total_points, dtype=bool)
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        in_E = np.zeros(total_points, dtype=bool)
        cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
        
        order = list(range(total_points))
        random.Random(42 + restart).shuffle(order)
        
        size = 0
        for z in order:
            if time.time() > deadline:
                break
            
            # Check if z can be added
            # Condition: exists v such that cnt[z, v] == q - 1
            # Wait, cnt[z, v] is the number of points in E on line(z, v) excluding z.
            # If cnt[z, v] == q - 1, then all other points are in E.
            
            if np.any(cnt[z, :] == q - 1):
                in_E[z] = True
                size += 1
                
                # Update counts for neighbors
                # For each v, neighbors are line_other[z, v]
                # Increment cnt[neighbor, v]
                for v in range(num_dirs):
                    neighbors = line_other[z, v, :]
                    cnt[neighbors, v] += 1
                    
        if size > best_size:
            best_size = size
            best_E = in_E.copy()

    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        # Fallback to trivial valid set (all points)
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)
