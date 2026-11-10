import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_points = q * q
    n_dirs = q + 1 # q slopes + infinity
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Elements are (a, b)
    
    # Precompute line members
    # For each point idx and direction d, list of other points on the line
    # idx = ((xa * p + xb) * q + ya) * p + yb
    
    def idx_to_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = idx // (p*p*p)
        return xa, xb, ya, yb

    def coords_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * q + ya) * p + yb

    # Directions: 0..q-1 are (1, m) with m = (m_a, m_b)
    # Direction q is (0, 1)
    
    # Precompute lines
    # lines[idx][d] = list of indices
    lines = np.zeros((n_points, n_dirs, q-1), dtype=np.int64)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = coords_to_idx(xa, xb, ya, yb)
                    for d in range(n_dirs):
                        if d < q:
                            ma = d // p
                            mb = d % p
                            # v = (1, m) = (1, 0) + (0, m) ? No, v = (1_a, 1_b, m_a, m_b)
                            # Point = (xa, xb) + t * (1, 0) + t * (0, m) ?
                            # v = (1, m). In F_49^2, v = (v_x, v_y).
                            # v_x = 1 = (1, 0). v_y = m = (ma, mb).
                            # x(t) = (xa, xb) + t * (1, 0)
                            # y(t) = (ya, yb) + t * (ma, mb)
                            
                            for t in range(1, q):
                                # t is in F_49. t = (ta, tb)
                                ta = t // p
                                tb = t % p
                                
                                # x(t) = (xa, xb) + (ta, tb)*(1, 0) = (xa + ta, xb)
                                tx_a = (xa + ta) % p
                                tx_b = xb
                                
                                # y(t) = (ya, yb) + (ta, tb)*(ma, mb)
                                # (ta, tb)*(ma, mb) = (ta*ma - tb*mb*k, ta*mb + tb*ma)
                                # k = 3
                                prod_a = (ta * ma - tb * mb * 3) % p
                                prod_b = (ta * mb + tb * ma) % p
                                ty_a = (ya + prod_a) % p
                                ty_b = (yb + prod_b) % p
                                
                                lines[idx, d, t-1] = coords_to_idx(tx_a, tx_b, ty_a, ty_b)
                        else:
                            # d = q, v = (0, 1) = (0, 0, 0, 1)
                            # x(t) = (xa, xb) + t*(0,0) = (xa, xb)
                            # y(t) = (ya, yb) + t*(0,1)
                            for t in range(1, q):
                                ta = t // p
                                tb = t % p
                                tx_a = xa
                                tx_b = xb
                                prod_a = (ta * 0 - tb * 1 * 3) % p # wait, m=(0,1)
                                # v_y = (0, 1). t * v_y = (ta, tb) * (0, 1)
                                # = (ta*0 - tb*1*3, ta*1 + tb*0) = (-3*tb, ta)
                                prod_a = (-3 * tb) % p
                                prod_b = ta % p
                                ty_a = (ya + prod_a) % p
                                ty_b = (yb + prod_b) % p
                                lines[idx, d, t-1] = coords_to_idx(tx_a, tx_b, ty_a, ty_b)
                                
    def greedy_run(seed):
        random.seed(seed)
        S = np.zeros(n_points, dtype=bool)
        C = np.zeros((n_points, n_dirs), dtype=np.int32)
        valid_count = np.full(n_points, n_dirs, dtype=np.int32)
        
        candidates = list(range(n_points))
        random.shuffle(candidates)
        
        for x in candidates:
            if S[x]:
                continue
            if valid_count[x] == 0:
                continue
                
            # Check safety
            safe = True
            # We need to check if any y in S loses its last valid direction
            # For each direction d, the points y affected are lines[x, d, :]
            # If y is in S, and C[y, d] == 0, and valid_count[y] == 1, then unsafe.
            # Note: if C[y, d] > 0, it's already blocked in that direction, so no change in valid_count.
            # We only care if C[y, d] transitions from 0 to 1.
            
            for d in range(n_dirs):
                # Get points on line through x with dir d
                ys = lines[x, d, :]
                for y in ys:
                    if S[y]:
                        if C[y, d] == 0:
                            if valid_count[y] == 1:
                                safe = False
                                break
                if not safe:
                    break
                    
            if safe:
                S[x] = True
                # Update counts
                for d in range(n_dirs):
                    ys = lines[x, d, :]
                    for y in ys:
                        if C[y, d] == 0:
                            valid_count[y] -= 1
                        C[y, d] += 1
                        
        return S

    start_time = time.time()
    best_S = None
    best_size = -1
    
    # Run multiple restarts
    # Estimate time per run
    # 2401 points * 50 dirs * 48 points = ~5.7M ops. Very fast in Python? Maybe 0.1-0.5s.
    # Let's try 100 restarts.
    
    for i in range(200):
        if time.time() - start_time > 500:
            break
        S = greedy_run(i)
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    if best_S is None:
        # Fallback to empty set (N = all points)
        best_S = np.zeros(n_points, dtype=bool)
        
    # Construct output
    # N is complement of S
    points = []
    for idx in range(n_points):
        if not best_S[idx]:
            xa, xb, ya, yb = idx_to_coords(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
