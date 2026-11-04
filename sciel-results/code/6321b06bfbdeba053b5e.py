"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 44
    # 16 for the 2x1x8 part
    # 4 * 7 = 28 for the four 2x2x2 parts
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_right (col 2) * B_bot (row 2)
    # A_right is 2x1, B_bot is 1x8.
    # Term: sum_{i,k} A[i, 2] * B[2, k] * C[i, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: A_left (cols 0,1) * B_top (rows 0,1)
    # B_top is 2x8. Split into 4 blocks of 2x2.
    # Strassen's algorithm for 2x2x2.
    
    # Define Strassen terms
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (local_i, local_j, coeff)
    # V_coeffs: list of (local_j, local_k, coeff)
    # W_coeffs: list of (local_i, local_k, coeff)
    
    strassen_terms = [
        # M0
        ([((0,0), 1), ((1,1), 1)], 
         [((0,0), 1), ((1,1), 1)], 
         [((0,0), 1), ((1,1), 1)]),
        # M1
        ([((1,0), 1), ((1,1), 1)], 
         [((0,0), 1)], 
         [((1,0), 1)]),
        # M2
        ([((0,0), 1)], 
         [((0,1), 1), ((1,1), -1)], 
         [((0,1), 1)]),
        # M3
        ([((1,1), 1)], 
         [((1,0), 1), ((0,0), -1)], 
         [((0,0), 1)]),
        # M4
        ([((0,0), 1), ((0,1), 1)], 
         [((1,1), 1)], 
         [((0,1), 1)]),
        # M5
        ([((1,0), 1), ((0,0), -1)], 
         [((0,0), 1), ((0,1), 1)], 
         [((0,0), -1), ((1,0), 1)]),
        # M6
        ([((0,1), 1), ((1,1), -1)], 
         [((1,0), 1), ((1,1), 1)], 
         [((0,0), 1), ((0,1), 1)]),
    ]
    
    # Wait, I need to double check the W coefficients for M5 and M6.
    # c00 = M0 + M3 - M5 + M6
    # c01 = M2 + M4
    # c10 = M1 + M5
    # c11 = M0 - M1 + M2 + M4
    
    # M5:
    # c00 coeff: -1
    # c10 coeff: 1
    # So W_coeffs for M5: [((0,0), -1), ((1,0), 1)]
    # My previous list had [((0,0), -1), ((1,0), 1)] for M5 W. Correct.
    
    # M6:
    # c00 coeff: 1
    # So W_coeffs for M6: [((0,0), 1)]
    # My previous list had [((0,0), 1), ((0,1), 1)] for M6 W. This is WRONG.
    # Let's re-verify c01. c01 = M2 + M4. M6 does not appear in c01.
    # So M6 W should only have c00.
    
    # Let's re-verify M5 W.
    # c00 = ... - M5 ...
    # c10 = M1 + M5
    # So M5 contributes to c00 and c10.
    # M5 W: [((0,0), -1), ((1,0), 1)]. Correct.
    
    # Let's re-verify M6 W.
    # c00 = ... + M6 ...
    # M6 does not appear in c01, c10, c11.
    # So M6 W: [((0,0), 1)].
    
    # Let's re-verify M2 W.
    # c01 = M2 + M4.
    # M2 W: [((0,1), 1)]. Correct.
    
    # Let's re-verify M4 W.
    # c01 = M2 + M4.
    # c11 = M0 - M1 + M2 + M4.
    # So M4 contributes to c01 and c11.
    # M4 W: [((0,1), 1), ((1,1), 1)].
    # My previous list for M4 W was [((0,1), 1)]. WRONG.
    
    # Let's re-verify M1 W.
    # c10 = M1 + M5.
    # c11 = M0 - M1 + M2 + M4.
    # So M1 contributes to c10 and c11.
    # M1 W: [((1,0), 1), ((1,1), -1)].
    # My previous list for M1 W was [((1,0), 1)]. WRONG.
    
    # Let's re-verify M3 W.
    # c00 = M0 + M3 - M5 + M6.
    # M3 does not appear elsewhere.
    # M3 W: [((0,0), 1)]. Correct.
    
    # Let's re-verify M0 W.
    # c00 = M0 ...
    # c11 = M0 ...
    # M0 W: [((0,0), 1), ((1,1), 1)]. Correct.

    # Corrected Strassen Terms:
    strassen_terms = [
        # M0
        ([((0,0), 1), ((1,1), 1)], 
         [((0,0), 1), ((1,1), 1)], 
         [((0,0), 1), ((1,1), 1)]),
        # M1
        ([((1,0), 1), ((1,1), 1)], 
         [((0,0), 1)], 
         [((1,0), 1), ((1,1), -1)]),
        # M2
        ([((0,0), 1)], 
         [((0,1), 1), ((1,1), -1)], 
         [((0,1), 1)]),
        # M3
        ([((1,1), 1)], 
         [((1,0), 1), ((0,0), -1)], 
         [((0,0), 1)]),
        # M4
        ([((0,0), 1), ((0,1), 1)], 
         [((1,1), 1)], 
         [((0,1), 1), ((1,1), 1)]),
        # M5
        ([((1,0), 1), ((0,0), -1)], 
         [((0,0), 1), ((0,1), 1)], 
         [((0,0), -1), ((1,0), 1)]),
        # M6
        ([((0,1), 1), ((1,1), -1)], 
         [((1,0), 1), ((1,1), 1)], 
         [((0,0), 1)]),
    ]

    for b in range(4):
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            # U
            for (li, lj), coeff in u_coeffs:
                # Local A(li, lj) -> Global A(li, lj)
                U[li * m + lj, col] += coeff
                
            # V
            for (lj, lk), coeff in v_coeffs:
                # Local B(lj, lk) -> Global B(lj, 2*b + lk)
                V[lj * p + (2 * b + lk), col] += coeff
                
            # W
            for (li, lk), coeff in w_coeffs:
                # Local C(li, lk) -> Global C(li, 2*b + lk)
                W[(2 * b + lk) * n + li, col] += coeff
                
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
