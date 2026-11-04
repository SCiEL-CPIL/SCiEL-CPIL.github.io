import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using 6 Strassen instances
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 6 block multiplications
    # (a_row_start, a_col_start, b_row_start, b_col_start)
    blocks = [
        (0, 0, 0, 0), # A11 B11
        (0, 2, 2, 0), # A12 B21
        (0, 0, 0, 2), # A11 B12
        (0, 2, 2, 2), # A12 B22
        (0, 0, 0, 4), # A11 B13
        (0, 2, 2, 4), # A12 B23
    ]
    
    # Strassen coefficients for X, Y
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22) Y11
    # M3 = X11 (Y12 - Y22)
    # M4 = X22 (Y21 - Y11)
    # M5 = (X11 + X12) Y22
    # M6 = (X21 - X11) (Y11 + Y12)
    # M7 = (X12 - X22) (Y21 + Y22)
    
    # X indices: 0:11, 1:12, 2:21, 3:22
    # Y indices: 0:11, 1:12, 2:21, 3:22
    
    strassen_x = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_y = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Z reconstruction
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    z_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # Z11
        [0, 0, 1, 0, 1, 0, 0], # Z12
        [0, 1, 0, 1, 0, 0, 0], # Z21
        [1, -1, 1, 0, 0, 1, 0]  # Z22
    ]
    
    for b_idx, (ar, ac, br, bc) in enumerate(blocks):
        for t in range(7):
            c = b_idx * 7 + t
            
            # Fill U
            # X is 2x2 block of A starting at ar, ac
            for xi in range(2):
                for xj in range(2):
                    # X index: xi*2 + xj
                    val = strassen_x[t][xi * 2 + xj]
                    if val != 0:
                        U[ar + xi][ac + xj][c] = val
                        # Note: U shape is (n*m, rank). Index is (ar+xi)*m + (ac+xj)?
                        # No, U index is i*m + j.
                        # A is 2x4. i in 0..1, j in 0..3.
                        # So index is (ar+xi)*4 + (ac+xj).
                        # Wait, U is (n*m, rank) = (8, 42).
                        # Row index is i*m + j.
                        # So U[(ar+xi)*m + (ac+xj)][c] = val.
                        pass
            # Correcting U index
            for xi in range(2):
                for xj in range(2):
                    val = strassen_x[t][xi * 2 + xj]
                    if val != 0:
                        u_idx = (ar + xi) * m + (ac + xj)
                        U[u_idx][c] = val
                        
            # Fill V
            # Y is 2x2 block of B starting at br, bc
            # B is 4x6. i in 0..3, j in 0..5.
            # V index is i*p + j? No, V is (m*p, rank) = (24, 42).
            # Row index is j*p + k? No, V corresponds to B.
            # B is 4x6. Rows 0..3, Cols 0..5.
            # V index is row*p + col?
            # The task says V is (m*p, rank). m=4, p=6.
            # So V index is j*p + k, where j is row of B (0..3), k is col of B (0..5).
            for yi in range(2):
                for yj in range(2):
                    val = strassen_y[t][yi * 2 + yj]
                    if val != 0:
                        v_idx = (br + yi) * p + (bc + yj)
                        V[v_idx][c] = val
                        
            # Fill W
            # Z is 2x2 block of C starting at ar, bc?
            # C is 2x6. Rows 0..1, Cols 0..5.
            # W index is k*n + i?
            # Task says W is (p*n, rank) = (12, 42).
            # W[k*n + i, c]. k is col of C (0..5), i is row of C (0..1).
            # The output block is at rows ar..ar+1, cols bc..bc+1.
            for zi in range(2):
                for zj in range(2):
                    val = z_coeffs[zi * 2 + zj][t]
                    if val != 0:
                        # C row is ar + zi
                        # C col is bc + zj
                        w_idx = (bc + zj) * n + (ar + zi)
                        W[w_idx][c] = val
                        
    return {"U": U, "V": V, "W": W}
