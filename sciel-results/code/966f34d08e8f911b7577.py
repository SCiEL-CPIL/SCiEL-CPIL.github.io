import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    NM, MP, PN = n * m, m * p, p * n
    
    U = [[0.0] * rank for _ in range(NM)]
    V = [[0.0] * rank for _ in range(MP)]
    W = [[0.0] * rank for _ in range(PN)]
    
    # Part 1: A_1 B_1 (14 terms)
    # A_1 is col 0 (j=0). B_1 is row 0 (m_row=0).
    r = 0
    for i in range(n):
        for k in range(p):
            # U index: i * m + 0
            U[i * m + 0][r] = 1.0
            # V index: 0 * p + k
            V[0 * p + k][r] = 1.0
            # W index: k * n + i
            W[k * n + i][r] = 1.0
            r += 1
            
    # Part 2: A_2 B_2 (25 terms)
    # A_2 is cols 1,2 (j=1,2). B_2 is rows 1,2 (m_row=1,2).
    # Blocks of B_2 columns:
    # Block 0: cols 0,1
    # Block 1: cols 2,3
    # Block 2: cols 4,5
    # Block 3: col 6
    
    strassen_coeffs = [
        # M1: (A00+A11)(B00+B11) -> C00(+), C11(+)
        ([(0,0), (1,1)], [(0,0), (1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B00 -> C10(+), C11(-)
        ([(1,0), (1,1)], [(0,0)], [(1,0,1), (1,1,-1)]),
        # M3: A00(B01-B11) -> C01(+), C11(+)
        ([(0,0)], [(0,1), (1,1)], [(0,1,1), (1,1,1)]),
        # M4: A11(B10-B00) -> C00(+), C10(+)
        ([(1,1)], [(1,0), (0,0)], [(0,0,1), (1,0,1)]),
        # M5: (A00+A01)B11 -> C00(-), C01(+)
        ([(0,0), (0,1)], [(1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10-A00)(B00+B01) -> C11(+)
        ([(1,0), (0,0)], [(0,0), (0,1)], [(1,1,1)]),
        # M7: (A01-A11)(B10+B11) -> C00(+), C01? No.
        # C00 = M1 + M4 - M5 + M7. So M7 contributes +1 to C00.
        # C11 = M1 - M2 + M3 + M6. M7 does not contribute to C11?
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Yes, M7 only in C00.
        ([(0,1), (1,1)], [(1,0), (1,1)], [(0,0,1)]),
    ]
    
    for b in range(3):
        k_start = 2 * b
        r_base = 14 + 7 * b
        for t in range(7):
            r = r_base + t
            u_elems, v_elems, w_elems = strassen_coeffs[t]
            
            for (i_loc, j_loc) in u_elems:
                # Global i = i_loc, j = 1 + j_loc
                U[i_loc * m + (1 + j_loc)][r] = 1.0
                
            for (k_loc, l_loc) in v_elems:
                # Global m_row = 1 + k_loc, col = k_start + l_loc
                V[(1 + k_loc) * p + (k_start + l_loc)][r] = 1.0
                
            for (i_out, k_out, coeff) in w_elems:
                # Global i = i_out, col = k_start + k_out
                W[(k_start + k_out) * n + i_out][r] = coeff

    # Block 3: 2x2x1 (4 terms)
    k_start = 6
    r_base = 14 + 21
    # Terms:
    # 1. A01 B16 -> C06
    # 2. A02 B26 -> C06
    # 3. A11 B16 -> C16
    # 4. A12 B26 -> C16
    
    r = r_base
    # 1
    U[0 * m + 1][r] = 1.0
    V[1 * p + 6][r] = 1.0
    W[6 * n + 0][r] = 1.0
    r += 1
    # 2
    U[0 * m + 2][r] = 1.0
    V[2 * p + 6][r] = 1.0
    W[6 * n + 0][r] = 1.0
    r += 1
    # 3
    U[1 * m + 1][r] = 1.0
    V[1 * p + 6][r] = 1.0
    W[6 * n + 1][r] = 1.0
    r += 1
    # 4
    U[1 * m + 2][r] = 1.0
    V[2 * p + 6][r] = 1.0
    W[6 * n + 1][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
