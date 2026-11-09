import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices: 0:00, 1:01, 2:10, 3:11
    # L_A[k][r][s] is a tuple of 4 coefficients for blocks A00, A01, A10, A11
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B01-B11)
    # k=3: M4 = A11(B10-B00)
    # k=4: M5 = (A00+A01)B11
    # k=5: M6 = (A10-A00)(B00+B01)
    # k=6: M7 = (A01-A11)(B10+B11)
    
    L_A = [
        [[(1,0,0,1)]*2 for _ in range(2)], # k=0
        [[(0,0,1,1)]*2 for _ in range(2)], # k=1
        [[(1,0,0,0)]*2 for _ in range(2)], # k=2
        [[(0,0,0,1)]*2 for _ in range(2)], # k=3
        [[(1,1,0,0)]*2 for _ in range(2)], # k=4
        [[(1,0,1,0)]*2 for _ in range(2)], # k=5 (A10-A00 -> 1 for A10, -1 for A00? Wait)
        [[(0,1,0,-1)]*2 for _ in range(2)]  # k=6 (A01-A11 -> 1 for A01, -1 for A11)
    ]
    # Correction for k=5: A10 - A00. Coeffs: A00:-1, A10:1.
    L_A[5] = [[(-1,0,1,0)]*2 for _ in range(2)]
    
    L_B = [
        [[(1,0,0,1)]*2 for _ in range(2)], # k=0
        [[(1,0,0,0)]*2 for _ in range(2)], # k=1
        [[(0,1,0,-1)]*2 for _ in range(2)], # k=2 (B01-B11)
        [[(0,0,1,-1)]*2 for _ in range(2)], # k=3 (B10-B00)
        [[(0,0,0,1)]*2 for _ in range(2)], # k=4
        [[(1,1,0,0)]*2 for _ in range(2)], # k=5 (B00+B01)
        [[(0,0,1,1)]*2 for _ in range(2)]  # k=6 (B10+B11)
    ]
    
    # Precompute base coefficients for U and V
    # U_base[k][r][s] is a list of 16 coefficients
    U_base = [[[None]*2 for _ in range(2)] for _ in range(7)]
    V_base = [[[None]*2 for _ in range(2)] for _ in range(7)]
    
    for k in range(7):
        for r in range(2):
            for s in range(2):
                u_vec = [0]*16
                v_vec = [0]*16
                for b in range(4):
                    coeff_a = L_A[k][r][s][b]
                    coeff_b = L_B[k][r][s][b]
                    if coeff_a != 0:
                        # Block b elements
                        row_off = 2 * (b // 2)
                        col_off = 2 * (b % 2)
                        for rr in range(2):
                            for ss in range(2):
                                idx = (row_off + rr) * 4 + (col_off + ss)
                                u_vec[idx] += coeff_a
                    if coeff_b != 0:
                        row_off = 2 * (b // 2)
                        col_off = 2 * (b % 2)
                        for rr in range(2):
                            for ss in range(2):
                                idx = (row_off + rr) * 4 + (col_off + ss)
                                v_vec[idx] += coeff_b
                U_base[k][r][s] = u_vec
                V_base[k][r][s] = v_vec
                
    # Inner Strassen combinations for U and V
    # u0: x00+x11, y00+y11
    # u1: x10+x11, y00
    # u2: x00, y01-y11
    # u3: x11, y10-y00
    # u4: x00+x01, y11
    # u5: x10-x00, y00+y01
    # u6: x01-x11, y10+y11
    
    U_combos = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    
    V_combos = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    
    # W coefficients
    # Z00 = u0 + u3 - u4 + u6
    # Z01 = u2 + u4
    # Z10 = u1 + u3
    # Z11 = u0 - u1 + u2 + u5
    
    W_u_coeffs = [
        [(0,0,1), (1,0,1), (0,1,-1), (1,1,1)], # Z00
        [(1,0,1), (0,1,1)], # Z01
        [(0,0,1), (1,0,1)], # Z10
        [(0,0,1), (1,0,-1), (1,1,1), (0,1,1)] # Z11
    ]
    
    # Block Strassen combinations for C
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    C_block_coeffs = [
        [(0,1), (3,1), (4,-1), (6,1)], # C00
        [(2,1), (4,1)], # C01
        [(1,1), (3,1)], # C10
        [(0,1), (1,-1), (2,1), (5,1)] # C11
    ]
    
    for k in range(7):
        for ell in range(7):
            c = k * 7 + ell
            
            # Fill U
            u_vec = [0]*16
            for r, s, coeff in U_combos[ell]:
                for i in range(16):
                    u_vec[i] += coeff * U_base[k][r][s][i]
            for i in range(16):
                U[i][c] = u_vec[i]
                
            # Fill V
            v_vec = [0]*16
            for r, s, coeff in V_combos[ell]:
                for i in range(16):
                    v_vec[i] += coeff * V_base[k][r][s][i]
            for i in range(16):
                V[i][c] = v_vec[i]
                
            # Fill W
            w_vec = [0]*16
            for r_prime, s_prime in [(0,0), (0,1), (1,0), (1,1)]:
                idx_z = r_prime * 2 + s_prime
                for I, J in [(0,0), (0,1), (1,0), (1,1)]:
                    idx_c = I * 2 + J
                    coeff_c = 0
                    for kk, coeff_k in C_block_coeffs[idx_c]:
                        if kk == k:
                            coeff_c += coeff_k
                    if coeff_c == 0:
                        continue
                    coeff_u = 0
                    for rr, ss, coeff_uu in W_u_coeffs[idx_z]:
                        if rr == ell:
                            coeff_u += coeff_uu
                    if coeff_u == 0:
                        continue
                    total_coeff = coeff_c * coeff_u
                    if total_coeff == 0:
                        continue
                    i = 2*I + r_prime
                    j = 2*J + s_prime
                    w_idx = j * 4 + i
                    w_vec[w_idx] += total_coeff
                    
            for i in range(16):
                W[i][c] = w_vec[i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
