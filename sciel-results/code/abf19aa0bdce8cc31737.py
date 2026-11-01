import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 groups * 14 terms per group = 56
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (Q00+Q11)(R00+R11)
    # M2 = (Q10+Q11)R00
    # M3 = Q00(R10-R11)
    # M4 = Q11(R00+R10)
    # M5 = (Q00+Q10)R11
    # M6 = (Q11-Q00)(R00+R01)
    # M7 = (Q01-Q11)(R10+R11)
    
    # Output reconstruction:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # Define Strassen steps
    # Each step is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i,j) in Q to coeff
    # V_coeffs: dict mapping (j,k) in R to coeff
    # W_coeffs: dict mapping (i,k) in P to coeff
    
    strassen_steps = [
        # M1
        ( {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1} ),
        # M2
        ( {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1} ), # Wait, P10=M2+M4, P11=M1-M2+M3+M6
        # Let's re-verify P indices for M2.
        # P10 = M2 + M4. So M2 contributes to P10 with +1.
        # P11 = M1 - M2 + M3 + M6. So M2 contributes to P11 with -1.
        # So W for M2 is {(1,0):1, (1,1):-1}. Correct.
        
        # M3
        ( {(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1} ), # P01=M3+M5, P11=M1-M2+M3+M6
        # So M3 contributes to P01 (+1) and P11 (+1).
        
        # M4
        ( {(1,1):1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1} ), # P00=M1+M4-M5+M7, P10=M2+M4
        # So M4 contributes to P00 (+1) and P10 (+1).
        
        # M5
        ( {(0,0):1, (1,0):1}, {(1,1):1}, {(0,0):-1, (0,1):1} ), # P00=M1+M4-M5+M7, P01=M3+M5
        # So M5 contributes to P00 (-1) and P01 (+1).
        
        # M6
        ( {(1,1):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1} ), # P11=M1-M2+M3+M6
        # So M6 contributes to P11 (+1).
        
        # M7
        ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1} ), # P00=M1+M4-M5+M7
        # So M7 contributes to P00 (+1).
    ]
    
    # We have 4 groups of columns in B (and C)
    # g in 0..3
    for g in range(4):
        # For each group, we compute C_g = A_L B_top_g + A_R B_bottom_g
        # A_L uses cols 0,1 of A. A_R uses cols 2,3 of A.
        # B_top_g uses rows 0,1 of B, cols 2g, 2g+1.
        # B_bottom_g uses rows 2,3 of B, cols 2g, 2g+1.
        
        # Process X = A_L B_top_g
        for u_c, v_c, w_c in strassen_steps:
            # Fill U, V, W for this rank-1 term
            for (qi, qj), coeff in u_c.items():
                # Q is A_L. qi in 0..1, qj in 0..1.
                # Global A index: qi * 4 + qj
                I = qi * 4 + qj
                U[I][col] += coeff
                
            for (rj, rk), coeff in v_c.items():
                # R is B_top_g. rj in 0..1 (local row), rk in 0..1 (local col).
                # Global B row: rj (since top is 0,1).
                # Global B col: 2g + rk.
                # Global B index: rj * 8 + (2g + rk)
                J = rj * 8 + (2 * g + rk)
                V[J][col] += coeff
                
            for (pi, pk), coeff in w_c.items():
                # P is C_g. pi in 0..1 (local row), pk in 0..1 (local col).
                # Global C col: 2g + pk.
                # Global C row: pi.
                # Global C index: pk * 2 + pi (trace convention k*n + i)
                K = (2 * g + pk) * 2 + pi
                W[K][col] += coeff
                
            col += 1
            
        # Process Y = A_R B_bottom_g
        for u_c, v_c, w_c in strassen_steps:
            for (qi, qj), coeff in u_c.items():
                # Q is A_R. qi in 0..1, qj in 0..1 (local).
                # Global A col: 2 + qj.
                I = qi * 4 + (2 + qj)
                U[I][col] += coeff
                
            for (rj, rk), coeff in v_c.items():
                # R is B_bottom_g. rj in 0..1 (local row), rk in 0..1 (local col).
                # Global B row: 2 + rj.
                # Global B col: 2g + rk.
                J = (2 + rj) * 8 + (2 * g + rk)
                V[J][col] += coeff
                
            for (pi, pk), coeff in w_c.items():
                # P is C_g. Same as above.
                K = (2 * g + pk) * 2 + pi
                W[K][col] += coeff
                
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
