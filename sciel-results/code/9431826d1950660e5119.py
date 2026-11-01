import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # A (2x4) = [A_L, A_R] where A_L, A_R are 2x2.
    # B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]] where blocks are 2x2.
    # C (2x4) = [C_L, C_R]
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    # Each 2x2 multiplication uses 7 multiplications (Strassen).
    # Total 4 products * 7 = 28 terms.
    
    rank = 28
    
    # Initialize factor matrices
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for X(2x2) * Y(2x2) = P(2x2)
    # X indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Y indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (x00+x11)(y00+y11) -> p00, p11
    # M2 = (x10+x11)y00 -> p10, p11
    # M3 = x00(y01-y11) -> p01, p11
    # M4 = x11(y10-y00) -> p00, p10
    # M5 = (x00+x01)y11 -> p00, p01
    # M6 = (x10-x00)(y00+y01) -> p11
    # M7 = (x01-x11)(y10+y11) -> p00
    
    # Each term is (u_indices, v_indices, w_indices)
    # u_indices: list of (local_idx, coeff)
    # v_indices: list of (local_idx, coeff)
    # w_indices: list of (local_idx, coeff)
    
    strassen_terms = [
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
    
    # Mapping functions to convert local 2x2 indices to global indices
    
    # A (2x4):
    # A_L: cols 0,1. Global indices:
    #   (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A_R: cols 2,3. Global indices:
    #   (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    
    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]
            
    # B (4x4):
    # B_TL: rows 0,1 cols 0,1. Global indices:
    #   (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B_TR: rows 0,1 cols 2,3. Global indices:
    #   (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # B_BL: rows 2,3 cols 0,1. Global indices:
    #   (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # B_BR: rows 2,3 cols 2,3. Global indices:
    #   (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    def map_v_b(block, local_idx):
        # local_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
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
    # W index is k*n + i = k*2 + i.
    # C_L: cols 0,1. k in {0,1}. i in {0,1}.
    #   local (r,c) -> global row r, col c.
    #   W index = c * 2 + r.
    # C_R: cols 2,3. k in {2,3}. i in {0,1}.
    #   local (r,c) -> global row r, col 2+c.
    #   W index = (2+c) * 2 + r.
    
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r
            
    # Apply Strassen to the 4 block products
    
    col = 0
    
    # 1. A_L * B_TL -> C_L
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('L', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('TL', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('L', idx), col] += coeff
        col += 1
        
    # 2. A_R * B_BL -> C_L
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('R', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('BL', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('L', idx), col] += coeff
        col += 1
            
    # 3. A_L * B_TR -> C_R
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('L', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('TR', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('R', idx), col] += coeff
        col += 1
            
    # 4. A_R * B_BR -> C_R
    for u_l, v_l, w_l in strassen_terms:
        for idx, coeff in u_l:
            U[map_u_a('R', idx), col] += coeff
        for idx, coeff in v_l:
            V[map_v_b('BR', idx), col] += coeff
        for idx, coeff in w_l:
            W[map_w_c('R', idx), col] += coeff
        col += 1
        
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
