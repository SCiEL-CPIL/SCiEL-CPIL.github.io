import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    deadline = time.time() + 570.0

    # Field arithmetic in F_49 = F_7[alpha], alpha^2 = 3
    k_const = 3
    
    elem_to_idx = {}
    idx_to_elem = []
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            elem_to_idx[(a, b)] = idx
            idx_to_elem.append((a, b))

    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        aa = (a1 * a2 + b1 * b2 * k_const) % p
        bb = (a1 * b2 + b1 * a2) % p
        return (aa, bb)

    # Directions: v = (0, 1) and v = (1, m) for m in F_49
    directions = []
    directions.append(((0, 0), (1, 0))) # (dx, dy)
    for m_idx in range(q):
        m_elem = idx_to_elem[m_idx]
        directions.append(((1, 0), m_elem))
        
    num_dirs = len(directions) # 50

    # Precompute line points for each (x, v)
    # line_pts[x][v] = list of q-1 point indices (excluding x)
    line_pts = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for pidx in range(total_points):
        xi = pidx // q
        yi = pidx % q
        x_elem = idx_to_elem[xi]
        y_elem = idx_to_elem[yi]
        for didx, (dx, dy) in enumerate(directions):
            count = 0
            for t_idx in range(q):
                if t_idx == 0:
                    continue
                t_elem = idx_to_elem[t_idx]
                tdx = f_mul(t_elem, dx)
                tdy = f_mul(t_elem, dy)
                nx = f_add(x_elem, tdx)
                ny = f_add(y_elem, tdy)
                nxi = elem_to_idx[nx]
                nyi = elem_to_idx[ny]
                npidx = nxi * q + nyi
                line_pts[pidx, didx, count] = npidx
                count += 1

    # Precompute affected: for each point z, list of (x, v) such that z is in line_pts[x][v]
    affected = [[] for _ in range(total_points)]
    for x in range(total_points):
        for v in range(num_dirs):
            for z in line_pts[x, v]:
                affected[z].append((x, v))

    # State:
    # in_N: bool array, True if point is in N
    in_N = np.zeros(total_points, dtype=bool)
    
    # cnt_not_in_N[x][v]: number of points in line_pts[x][v] that are NOT in N
    # Initially all are not in N, so count is q-1 = 48
    cnt_not_in_N = np.full((total_points, num_dirs), q - 1, dtype=np.int32)
    
    # valid[x]: True if min(cnt_not_in_N[x]) == 0
    # We need all x to be valid.
    
    # Helper: check if x is valid
    def is_valid(x):
        return np.min(cnt_not_in_N[x]) == 0

    # Find all currently invalid points
    # invalid_mask[x] = not is_valid(x)
    
    # Greedy Loop
    # We need to add points to N until all are valid.
    # Strategy:
    # 1. Find an invalid point x.
    # 2. For that x, find the direction v that minimizes cnt_not_in_N[x][v].
    # 3. Add all points in line_pts[x][v] that are not in N to N.
    # 4. Update counts.
    
    # To speed up finding "an invalid point", we can maintain a list or just scan.
    # Scanning 2401 points is fast.
    
    while True:
        if time.time() > deadline:
            break
            
        # Find first invalid point
        invalid_x = -1
        for x in range(total_points):
            if not is_valid(x):
                invalid_x = x
                break
                
        if invalid_x == -1:
            # All valid
            break
            
        # For invalid_x, find best direction v
        best_v = -1
        min_cnt = q
        for v in range(num_dirs):
            if cnt_not_in_N[invalid_x, v] < min_cnt:
                min_cnt = cnt_not_in_N[invalid_x, v]
                best_v = v
                
        # Add points from line_pts[invalid_x][best_v] to N
        # Only those not already in N
        for z in line_pts[invalid_x, best_v]:
            if not in_N[z]:
                in_N[z] = True
                # Update cnt_not_in_N for all (x, v) affected by z
                for (x, v) in affected[z]:
                    cnt_not_in_N[x, v] -= 1
                    
    # Build output N
    points_list = []
    for pidx in range(total_points):
        if in_N[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])
            
    if not points_list:
        # Fallback
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])
            
    return np.array(points_list, dtype=np.int64)
