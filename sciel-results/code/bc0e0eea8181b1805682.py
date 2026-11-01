import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank of M_{2,4,4} is 28.
    # We construct the rank-28 decomposition using Strassen's algorithm
    # applied to the four 2x2 block products.
    # A (2x4) = [A_L, A_R]
    # B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]]
    # C (2x4) = [C_L, C_R]
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 terms
    # Local indices for 2x2 matrix: 0:00, 1:01, 2:10, 3:11
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),              # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),              # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),              # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),              # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),              # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),              # M7
    ]
    
    # Mapping functions
    # A (2x4):
    #   A_L: cols 0,1 -> indices 0,1,4,5 (local 0,1,2,3 -> global 0,1,4,5)
    #   A_R: cols 2,3 -> indices 2,3,6,7 (local 0,1,2,3 -> global 2,3,6,7)
    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]
            
    # B (4x4):
    #   B_TL: rows 0,1 cols 0,1 -> indices 0,1,4,5
    #   B_TR: rows 0,1 cols 2,3 -> indices 2,3,6,7
    #   B_BL: rows 2,3 cols 0,1 -> indices 8,9,12,13
    #   B_BR: rows 2,3 cols 2,3 -> indices 10,11,14,15
    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)
            
    # C (2x4):
    #   W index is k*n + i = k*2 + i.
    #   C_L: cols 0,1. k in {0,1}.
    #   C_R: cols 2,3. k in {2,3}.
    #   Local (r,c) in 2x2 result: r is row (i), c is col (k).
    #   Global col c is the 'k' index. Global row r is the 'i' index.
    #   W index = k*2 + i.
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            k = c
        else:
            k = 2 + c
        return k * 2 + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        col = start_col
        for u_local, v_local, w_local in strassen_terms:
            for idx, coeff in u_local:
                U[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w_c(c_side, idx), col] += coeff
            col += 1
        return col

    # Apply Strassen to the 4 block products
    # 1. A_L * B_TL -> C_L
    col = apply_strassen('L', 'TL', 'L', 0)
    # 2. A_R * B_BL -> C_L
    col = apply_strassen('R', 'BL', 'L', col)
    # 3. A_L * B_TR -> C_R
    col = apply_strassen('L', 'TR', 'R', col)
    # 4. A_R * B_BR -> C_R
    col = apply_strassen('R', 'BR', 'R', col)
    
    # The rank is exactly 28.
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
