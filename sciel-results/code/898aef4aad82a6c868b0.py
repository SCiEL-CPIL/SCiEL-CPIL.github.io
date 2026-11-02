import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # Using 4 Strassen decompositions for 2x2 blocks
    # Total rank = 4 * 7 = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2 mult coefficients
    # M1 = (q00+q11)(r00+r11)
    # M2 = (q10+q11)r00
    # M3 = q00(r01-r11)
    # M4 = q11(r10-r00)
    # M5 = (q00+q01)r11
    # M6 = (q10-q00)(r00+r01)
    # M7 = (q01-q11)(r10+r11)
    
    # Output reconstruction:
    # P00 = M1 + M4 - M6 + M7
    # P01 = M3 + M5 + M6
    # P10 = M2 + M4 + M7
    # P11 = M1 - M3 + M5 - M2
    
    strassen_u_coeffs = [
        [1, 1],   # M1: q00, q11
        [0, 1],   # M2: q10, q11 (Wait, M2 uses q10 and q11? No. M2 = (q10+q11)r00. So q10, q11)
        [1, 0],   # M3: q00
        [0, 1],   # M4: q11
        [1, 1],   # M5: q00, q01
        [1, -1],  # M6: q10, q00 (Wait, M6=(q10-q00)... so q10, q00 with signs)
        [0, 1]    # M7: q01, q11 (Wait, M7=(q01-q11)... so q01, q11)
    ]
    # Let's be precise with indices in the 2x2 Q matrix
    # Q = [[q00, q01], [q10, q11]]
    
    # M1: (q00+q11) -> coeffs for [q00, q01, q10, q11] are [1, 0, 0, 1]
    # M2: (q10+q11) -> [0, 0, 1, 1]
    # M3: q00 -> [1, 0, 0, 0]
    # M4: q11 -> [0, 0, 0, 1]
    # M5: (q00+q01) -> [1, 1, 0, 0]
    # M6: (q10-q00) -> [-1, 0, 1, 0]
    # M7: (q01-q11) -> [0, 1, 0, -1]
    
    q_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # R = [[r00, r01], [r10, r11]]
    # M1: (r00+r11) -> [1, 0, 0, 1]
    # M2: r00 -> [1, 0, 0, 0]
    # M3: (r01-r11) -> [0, 1, 0, -1]
    # M4: (r10-r00) -> [-1, 0, 1, 0]
    # M5: r11 -> [0, 0, 0, 1]
    # M6: (r00+r01) -> [1, 1, 0, 0]
    # M7: (r10+r11) -> [0, 0, 1, 1]
    
    r_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Output P coeffs for M1..M7
    # P00 = M1 + M4 - M6 + M7
    # P01 = M3 + M5 + M6
    # P10 = M2 + M4 + M7
    # P11 = M1 - M3 + M5 - M2
    # Order: [M1, M2, M3, M4, M5, M6, M7]
    
    p_coeffs = [
        [1, 0, 0, 1, 0, -1, 1],  # P00
        [0, 0, 1, 0, 1, 1, 0],   # P01
        [0, 1, 0, 1, 0, 0, 1],   # P10
        [1, -1, -1, 0, 1, 0, 0]  # P11
    ]
    
    # Define the 4 blocks
    # Block 1: X1 = A1 B11. A1 rows 0,1 cols 0,1. B11 rows 0,1 cols 0,1.
    # Block 2: X2 = A2 B21. A2 rows 0,1 cols 2,3. B21 rows 2,3 cols 0,1.
    # Block 3: X3 = A1 B12. A1 rows 0,1 cols 0,1. B12 rows 0,1 cols 2,3.
    # Block 4: X4 = A2 B22. A2 rows 0,1 cols 2,3. B22 rows 2,3 cols 2,3.
    
    # A indices in U: i*m + j
    # B indices in V: j*p + k
    # C indices in W: k*n + i
    
    blocks = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        # A is 2x4. B is 4x4. C is 2x4.
        # X1 = A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 0, 0, 0),
        # X2 = A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0, 0, 0),
        # X3 = A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 0, 2, 0, 2),
        # X4 = A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 2, 0, 2),
    ]
    
    col = 0
    for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        for s in range(7):
            # Fill U
            for qi in range(2):
                for qj in range(2):
                    coeff = q_coeffs[s][qi*2 + qj]
                    if coeff != 0:
                        u_idx = (ar + qi) * m + (ac + qj)
                        U[u_idx][col] += coeff
                        
            # Fill V
            for ri in range(2):
                for rj in range(2):
                    coeff = r_coeffs[s][ri*2 + rj]
                    if coeff != 0:
                        v_idx = (br + ri) * p + (bc + rj)
                        V[v_idx][col] += coeff
                        
            # Fill W
            # The result of this Strassen step contributes to P00..P11
            # Which map to C elements at (cr+i, cc+j)
            # P00 -> C[cr+0, cc+0]
            # P01 -> C[cr+0, cc+1]
            # P10 -> C[cr+1, cc+0]
            # P11 -> C[cr+1, cc+1]
            
            # p_coeffs[pi] is the coeffs for M1..M7 for P_i
            # We need to add p_coeffs[pi][s] to W at index (cc+j)*n + (cr+i)
            # Note: W index is k*n + i. Here k is col index in C, i is row index in C.
            # C row index = cr + i. C col index = cc + j.
            # So W index = (cc + j) * n + (cr + i)
            
            for pi in range(4):
                coeff = p_coeffs[pi][s]
                if coeff != 0:
                    # pi=0: i=0, j=0
                    # pi=1: i=0, j=1
                    # pi=2: i=1, j=0
                    # pi=3: i=1, j=1
                    pi_row, pi_col = divmod(pi, 2) # This is wrong mapping. 
                    # P is 2x2. P00 (row 0 col 0), P01 (row 0 col 1), P10 (row 1 col 0), P11 (row 1 col 1).
                    # My p_coeffs list order: [P00, P01, P10, P11].
                    # So pi=0 -> r=0, c=0. pi=1 -> r=0, c=1. pi=2 -> r=1, c=0. pi=3 -> r=1, c=1.
                    pr = pi // 2
                    pc = pi % 2
                    w_idx = (cc + pc) * n + (cr + pr)
                    W[w_idx][col] += coeff
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
