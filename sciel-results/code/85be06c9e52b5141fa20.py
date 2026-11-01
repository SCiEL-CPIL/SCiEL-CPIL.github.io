import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin

    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_indices, v_indices, w_indices, col_idx):
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff

    # Strassen 2x2 terms
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(left_right, local_idx):
        if left_right == 'L':
            return local_idx
        else:
            return local_idx + 2

    def map_v_b(block, local_idx):
        # block: 'TL', 'TR', 'BL', 'BR'
        # local_idx: 0..3 (r=0..1, c=0..1)
        r = local_idx // 2
        c = local_idx % 2
        
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (c + 2)
        elif block == 'BL':
            return (r + 2) * 4 + c
        elif block == 'BR':
            return (r + 2) * 4 + (c + 2)

    def map_w_c(left_right, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if left_right == 'L':
            return c * 2 + r
        else:
            return (c + 2) * 2 + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        if time.time() > deadline:
            return start_col
        for t in strassen_terms:
            u_local, v_local, w_local = t
            u_global = []
            v_global = []
            w_global = []
            
            for idx, coeff in u_local:
                g_idx = map_u_a(a_side, idx)
                u_global.append((g_idx, coeff))
            
            for idx, coeff in v_local:
                g_idx = map_v_b(b_block, idx)
                v_global.append((g_idx, coeff))
            
            for idx, coeff in w_local:
                g_idx = map_w_c(c_side, idx)
                w_global.append((g_idx, coeff))
            
            add_term(u_global, v_global, w_global, start_col)
            start_col += 1
        
        return start_col

    # 1. A_L * B_TL -> C_L
    col = apply_strassen('L', 'TL', 'L', 0)
    
    # 2. A_R * B_BL -> C_L
    col = apply_strassen('R', 'BL', 'L', col)
    
    # 3. A_L * B_TR -> C_R
    col = apply_strassen('L', 'TR', 'R', col)
    
    # 4. A_R * B_BR -> C_R
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
