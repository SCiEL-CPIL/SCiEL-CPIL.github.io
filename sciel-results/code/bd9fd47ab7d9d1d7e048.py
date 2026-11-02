import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
# Pre-compute Strassen terms and mapping helpers to reduce overhead in construct()
_STRASSEN_TERMS = [
    # M1
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    # M2
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    # M3
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    # M4
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
    # M5
    ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
    # M6
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
    # M7
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
]

def _map_u_a(left_right, local_idx):
    if left_right == 'L':
        if local_idx == 0: return 0
        if local_idx == 1: return 1
        if local_idx == 2: return 4
        if local_idx == 3: return 5
    else:
        if local_idx == 0: return 2
        if local_idx == 1: return 3
        if local_idx == 2: return 6
        if local_idx == 3: return 7

def _map_v_b(block, local_idx):
    if block == 'TL':
        row_off = 0; col_off = 0
    elif block == 'TR':
        row_off = 0; col_off = 2
    elif block == 'BL':
        row_off = 2; col_off = 0
    elif block == 'BR':
        row_off = 2; col_off = 2
    else:
        return 0
        
    r = local_idx // 2
    c = local_idx % 2
    global_r = row_off + r
    global_c = col_off + c
    return global_r * 4 + global_c

def _map_w_c(left_right, local_idx):
    r = local_idx // 2
    c = local_idx % 2
    if left_right == 'L':
        global_c = c
    else:
        global_c = 2 + c
    return global_c * 2 + r

def construct():
    _t0 = time.time()
    _deadline = 110.0
    n, m, p = DIMENSIONS
    
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    if time.time() - _t0 > _deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def add_term(u_indices, v_indices, w_indices, col_idx):
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff

    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in _STRASSEN_TERMS:
            u_local, v_local, w_local = t
            u_global = []
            v_global = []
            w_global = []
            
            for idx, coeff in u_local:
                g_idx = _map_u_a(a_side, idx)
                u_global.append((g_idx, coeff))
            
            for idx, coeff in v_local:
                g_idx = _map_v_b(b_block, idx)
                v_global.append((g_idx, coeff))
            
            for idx, coeff in w_local:
                g_idx = _map_w_c(c_side, idx)
                w_global.append((g_idx, coeff))
            
            add_term(u_global, v_global, w_global, start_col)
            start_col += 1
        
        return start_col

    col = apply_strassen('L', 'TL', 'L', 0)
    
    if time.time() - _t0 > _deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    col = apply_strassen('R', 'BL', 'L', col)
    
    if time.time() - _t0 > _deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    col = apply_strassen('L', 'TR', 'R', col)
    
    if time.time() - _t0 > _deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    col = apply_strassen('R', 'BR', 'R', col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
