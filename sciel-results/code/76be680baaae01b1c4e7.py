"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    The decomposition is based on applying Strassen's algorithm to two 2x2 blocks
    of the 2x4 matrix B.
    """
    n, m, p = DIMENSIONS
    rank = 14
    
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2): 
    # 0: a00, 1: a01, 2: a10, 3: a11
    
    # Indices for B (2x4):
    # Row 0: 0: b00, 1: b01, 2: b02, 3: b03
    # Row 1: 4: b10, 5: b11, 6: b12, 7: b13
    
    # Indices for C (2x4) in W (k*n + i where k is col, i is row):
    # Col 0: 0: c00, 1: c10
    # Col 1: 2: c01, 3: c11
    # Col 2: 4: c02, 5: c12
    # Col 3: 6: c03, 7: c13
    
    # We split B into B1 (cols 0,1) and B2 (cols 2,3).
    # Each block multiplication (2x2 * 2x2) uses 7 Strassen products.
    # Total rank = 7 + 7 = 14.
    
    # --- Block 1: C1 = A * B1 ---
    # B1 indices in V: 0: b00, 1: b01, 4: b10, 5: b11
    
    # M1: (a00 + a11) * (b00 + b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2: (a10 + a11) * b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 -= M2
    W[1][1] = 1
    W[3][1] = -1
    
    # M3: a00 * (b01 - b11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    # C01 += M3, C11 += M3
    W[2][2] = 1
    W[3][2] = 1
    
    # M4: a11 * (b10 - b00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    # C00 += M4, C10 += M4
    W[0][3] = 1
    W[1][3] = 1
    
    # M5: (a00 + a01) * b11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    # C02? No, C01? 
    # Strassen: C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # Let's re-verify standard Strassen output mapping for 2x2.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # So for Block 1 (Cols 0,1 of C):
    # C00 (W[0]): M1 + M4 - M5 + M7
    # C10 (W[1]): M2 + M4
    # C01 (W[2]): M3 + M5
    # C11 (W[3]): M1 - M2 + M3 + M6
    
    # We already set:
    # W[0][0] = 1 (M1)
    # W[3][0] = 1 (M1)
    # W[1][1] = 1 (M2)
    # W[3][1] = -1 (M2)
    # W[2][2] = 1 (M3)
    # W[3][2] = 1 (M3)
    # W[0][3] = 1 (M4)
    # W[1][3] = 1 (M4)
    
    # Now M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    # C00 -= M5, C01 += M5
    W[0][4] = -1
    W[2][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 += M6
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    # C00 += M7
    W[0][6] = 1

    # --- Block 2: C2 = A * B2 ---
    # B2 indices in V: 2: b02, 3: b03, 6: b12, 7: b13
    # Same A forms, so U is identical to Block 1 but shifted by 7 in rank index.
    # Rank indices 7-13.
    
    # N1: (a00 + a11) * (b02 + b13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    # C02 (W[4]) += N1, C12 (W[7]) += N1
    W[4][7] = 1
    W[7][7] = 1
    
    # N2: (a10 + a11) * b02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    # C12 (W[5]) += N2, C13 (W[7]) -= N2
    W[5][8] = 1
    W[7][8] = -1
    
    # N3: a00 * (b03 - b13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    # C03 (W[6]) += N3, C13 (W[7]) += N3
    W[6][9] = 1
    W[7][9] = 1
    
    # N4: a11 * (b12 - b02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    # C02 (W[4]) += N4, C12 (W[5]) += N4
    W[4][10] = 1
    W[5][10] = 1
    
    # N5: (a00 + a01) * b13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    # C02 (W[4]) -= N5, C03 (W[6]) += N5
    W[4][11] = -1
    W[6][11] = 1
    
    # N6: (a10 - a00) * (b02 + b03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    # C13 (W[7]) += N6
    W[7][12] = 1
    
    # N7: (a01 - a11) * (b12 + b13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    # C02 (W[4]) += N7
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
