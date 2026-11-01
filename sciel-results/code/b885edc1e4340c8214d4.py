import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition components
    # u_s, v_s are 4-vectors (flattened 2x2)
    # w_s are 2x2 matrices
    # Order: p00, p01, p10, p11
    strassen_u = [
        [1, 0, 0, 0],      # s=1: p00
        [0, 0, 1, 1],      # s=2: p10 + p11
        [1, 0, 0, 0],      # s=3: p00
        [0, 0, 0, 1],      # s=4: p11
        [1, 1, 0, 0],      # s=5: p00 + p01
        [-1, 0, 1, 0],     # s=6: -p00 + p10
        [0, 1, -1, 0]      # s=7: p01 - p11
    ]
    strassen_v = [
        [1, 0, 0, 0],      # s=1: q00
        [1, 0, 0, 0],      # s=2: q00
        [0, 1, 0, -1],     # s=3: q01 - q11
        [-1, 0, 1, 0],     # s=4: -q00 + q10
        [0, 0, 0, 1],      # s=5: q11
        [1, 1, 0, 0],      # s=6: q00 + q01
        [0, 0, 1, 1]       # s=7: q10 + q11
    ]
    strassen_w = [
        [[1, 0], [0, 1]],  # s=1: I
        [[0, 0], [1, -1]], # s=2
        [[0, 1], [0, 1]],  # s=3
        [[1, 0], [1, 0]],  # s=4
        [[-1, 1], [0, 0]], # s=5
        [[0, 0], [0, 1]],  # s=6
        [[1, 0], [0, 0]]   # s=7
    ]
    
    # Block combinations for Strassen 4x4
    # M_k = L_k(A) R_k(B)
    # L_k coefficients for A00, A01, A10, A11
    # R_k coefficients for B00, B01, B10, B11
    # Order: 00, 01, 10, 11
    block_L = [
        [1, 0, 0, 0],      # M1: A00
        [0, 0, 1, 1],      # M2: A10 + A11
        [1, 0, 0, 0],      # M3: A00
        [0, 0, 0, 1],      # M4: A11
        [1, 1, 0, 0],      # M5: A00 + A01
        [-1, 0, 1, 0],     # M6: -A00 + A10
        [0, 1, 0, -1]      # M7: A01 - A11
    ]
    block_R = [
        [1, 0, 0, 0],      # M1: B00
        [1, 0, 0, 0],      # M2: B00
        [0, 1, 0, -1],     # M3: B01 - B11
        [-1, 0, 1, 0],     # M4: -B00 + B10
        [0, 0, 0, 1],      # M5: B11
        [1, 1, 0, 0],      # M6: B00 + B01
        [0, 0, 1, 1]       # M7: B10 + B11
    ]
    
    # Output combination coefficients alpha_KL_k
    # C_KL = sum_k alpha_KL_k M_k
    # K, L in 0..1
    # alpha[K][L][k]
    alpha = [
        [
            [1, 0, 0, 1, -1, 0, 1],   # C00: M1 + M4 - M5 + M7
            [0, 1, 0, 1, 0, 0, 0]     # C10: M2 + M4
        ],
        [
            [0, 0, 1, 0, 1, 0, 0],    # C01: M3 + M5
            [1, -1, 1, 0, 0, 1, 0]    # C11: M1 - M2 + M3 + M6
        ]
    ]
    # Note: alpha is indexed [K][L][k] but above C10 is K=1, L=0?
    # C_KL where K is row block, L is col block.
    # C00 (K=0, L=0), C01 (K=0, L=1), C10 (K=1, L=0), C11 (K=1, L=1)
    # Let's re-verify indices.
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Correct alpha structure: alpha[K][L][k]
    alpha = [
        [
            [1, 0, 0, 1, -1, 0, 1],   # C00 (K=0, L=0)
            [0, 0, 1, 0, 1, 0, 0]     # C01 (K=0, L=1)
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],    # C10 (K=1, L=0)
            [1, -1, 1, 0, 0, 1, 0]    # C11 (K=1, L=1)
        ]
    ]
    
    col = 0
    for k in range(7):
        for s in range(7):
            # Calculate linear forms for U and V
            # u_s(L_k(A))
            # L_k(A) is a 2x2 matrix P where P_pq = sum_b block_L[k][b] A_block_b_pq
            # u_s is a vector [u0, u1, u2, u3] corresponding to P00, P01, P10, P11
            # Coeff for A_ij (global)
            # A_block_b_pq corresponds to global A_{2I+pq_row, 2J+pq_col}
            
            # For each element of the 2x2 P matrix (index r, c in 0..1)
            # P_rc = sum_b block_L[k][b] A_{2I_b + r, 2J_b + c}
            # u_s contribution to term is u_s[r*2+c] * P_rc
            # So coeff for A_{2I_b + r, 2J_b + c} is u_s[r*2+c] * block_L[k][b]
            
            u_coeffs = [0] * 16
            for r in range(2):
                for c in range(2):
                    u_val = strassen_u[s][r * 2 + c]
                    if u_val == 0: continue
                    for b in range(4):
                        # b: 0->A00, 1->A01, 2->A10, 3->A11
                        I = b // 2
                        J = b % 2
                        coeff = u_val * block_L[k][b]
                        if coeff != 0:
                            i_global = 2 * I + r
                            j_global = 2 * J + c
                            u_coeffs[i_global * 4 + j_global] = coeff

            v_coeffs = [0] * 16
            for r in range(2):
                for c in range(2):
                    v_val = strassen_v[s][r * 2 + c]
                    if v_val == 0: continue
                    for b in range(4):
                        # b: 0->B00, 1->B01, 2->B10, 3->B11
                        I = b // 2
                        J = b % 2
                        coeff = v_val * block_R[k][b]
                        if coeff != 0:
                            i_global = 2 * I + r
                            j_global = 2 * J + c
                            v_coeffs[i_global * 4 + j_global] = coeff

            # Determine W contributions
            # w_s is 2x2 matrix W_s
            # Contributes to M_k (block)
            # M_k is a 2x2 matrix.
            # C_KL = sum_k alpha[K][L][k] M_k
            # So contribution to C_KL element (p, q) is alpha[K][L][k] * w_s[p][q]
            
            w_matrix = strassen_w[s]
            
            for K in range(2):
                for L in range(2):
                    a_val = alpha[K][L][k]
                    if a_val == 0: continue
                    for p in range(2):
                        for q in range(2):
                            w_val = w_matrix[p][q]
                            if w_val == 0: continue
                            coeff = a_val * w_val
                            if coeff == 0: continue
                            
                            # Output index: k_out * 4 + i_out
                            # k_out = 2L + q
                            # i_out = 2K + p
                            k_out = 2 * L + q
                            i_out = 2 * K + p
                            w_idx = k_out * 4 + i_out
                            
                            W[w_idx][col] += coeff
            
            # Add U and V coeffs
            for i in range(16):
                U[i][col] = u_coeffs[i]
                V[i][col] = v_coeffs[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
