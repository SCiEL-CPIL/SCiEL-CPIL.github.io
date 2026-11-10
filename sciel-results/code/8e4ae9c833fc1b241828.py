import time
import numpy as np
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 615.0

p = 7
K = 3
N_TOTAL = p**4
N_PER_LINE = p*p
N_LINES_TOTAL = 49 + 2401

def _precompute_field():
    add = np.zeros((49, 49), dtype=np.int32)
    mul = np.zeros((49, 49), dtype=np.int32)
    inv = np.zeros(49, dtype=np.int32)
    
    for a1 in range(7):
        for b1 in range(7):
            v1 = a1 * 7 + b1
            for a2 in range(7):
                for b2 in range(7):
                    v2 = a2 * 7 + b2
                    ra = (a1 + a2) % 7
                    rb = (b1 + b2) % 7
                    add[v1, v2] = ra * 7 + rb
                    
                    ra = (a1 * a2 + 3 * b1 * b2) % 7
                    rb = (a1 * b2 + a2 * b1) % 7
                    mul[v1, v2] = ra * 7 + rb
    
    for v in range(1, 49):
        for w in range(1, 49):
            if mul[v, w] == 0:
                inv[v] = w
                break
    inv[0] = 0
    return add, mul, inv

ADD, MUL, INV = _precompute_field()

def _field_sub(x, y):
    ya, yb = divmod(y, 7)
    na = (7 - ya) % 7
    nb = (7 - yb) % 7
    return ADD[x, na * 7 + nb]

def _field_mul(x, y):
    return MUL[x, y]

def _field_inv(x):
    return INV[x]

def _precompute_points():
    point_u = np.zeros(N_TOTAL, dtype=np.int32)
    point_v = np.zeros(N_TOTAL, dtype=np.int32)
    for idx in range(N_TOTAL):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        point_u[idx] = xa * 7 + xb
        point_v[idx] = ya * 7 + yb
    return point_u, point_v

def _precompute_lines_through(point_u, point_v):
    lines_through = np.zeros((N_TOTAL, 50), dtype=np.int32)
    for idx in range(N_TOTAL):
        u = point_u[idx]
        v = point_v[idx]
        lines_through[idx, 0] = u
        for m in range(49):
            c = _field_sub(v, _field_mul(m, u))
            lines_through[idx, m + 1] = 49 + m * 49 + c
    return lines_through

def nikodym_set() -> np.ndarray:
    point_u, point_v = _precompute_points()
    lines_through = _precompute_lines_through(point_u, point_v)
    
    in_S = np.zeros(N_TOTAL, dtype=np.int8)
    line_counts = np.zeros(N_LINES_TOTAL, dtype=np.int32)
    is_good = np.zeros(N_TOTAL, dtype=np.int8)
    
    bad_count = 0
    for x in range(N_TOTAL):
        is_good[x] = 1
        
    best_bad = bad_count
    best_in_S = in_S.copy()
    
    random.seed(42)
    T = 1.0
    
    while time.time() < _DEADLINE - 10:
        y = random.randint(0, N_TOTAL - 1)
        
        old_bad = bad_count
        old_in_S_y = in_S[y]
        old_is_good = is_good.copy()
        old_line_counts = line_counts.copy()
        
        in_S[y] = 1 - old_in_S_y
        
        for lid in lines_through[y]:
            line_counts[lid] += 1 if in_S[y] else -1
            
        new_bad = 0
        for x in range(N_TOTAL):
            l_ids = lines_through[x]
            cnts = line_counts[l_ids]
            m1 = np.min(cnts)
            
            if (in_S[x] and m1 == 1) or (not in_S[x] and m1 == 0):
                new_good = 1
            else:
                new_good = 0
                
            if new_good != is_good[x]:
                if new_good:
                    new_bad -= 1
                else:
                    new_bad += 1
                is_good[x] = new_good
                
        bad_count = new_bad
        
        if bad_count < old_bad:
            if bad_count < best_bad:
                best_bad = bad_count
                best_in_S = in_S.copy()
        else:
            # Reject and rollback
            in_S[y] = old_in_S_y
            line_counts[:] = old_line_counts
            is_good[:] = old_is_good
            bad_count = old_bad
            
        # Anneal
        T *= 0.999
        if T < 0.1:
            T = 0.1

    points = []
    for idx in range(N_TOTAL):
        if not best_in_S[idx]:
            xa, xb = divmod(point_u[idx], 7)
            ya, yb = divmod(point_v[idx], 7)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
