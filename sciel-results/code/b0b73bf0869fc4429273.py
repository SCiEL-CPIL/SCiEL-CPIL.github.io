import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22) Y11
    # M3 = X11 (Y12 - Y22)
    # M4 = X22 (Y21 - Y11)
    # M5 = (X11 + X12) Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # U indices for X (2x2):
    # X11 -> 0, X12 -> 1, X21 -> 2, X22 -> 3
    # V indices for Y (2x2):
    # Y11 -> 0, Y12 -> 1, Y21 -> 2, Y22 -> 3
    # W indices for Z (2x2):
    # Z11 -> 0, Z12 -> 1, Z21 -> 2, Z22 -> 3
    
    # Coefficients for U (X)
    # M1: X11=1, X22=1
    # M2: X21=1, X22=1
    # M3: X11=1
    # M4: X22=1
    # M5: X11=1, X12=1
    # M6: X21=1, X11=-1
    # M7: X12=1, X22=-1
    
    u_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Coefficients for V (Y)
    # M1: Y11=1, Y22=1
    # M2: Y11=1
    # M3: Y12=1, Y22=-1
    # M4: Y21=1, Y11=-1
    # M5: Y22=1
    # M6: Y11=1, Y12=1
    # M7: Y21=1, Y22=1
    
    v_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for W (Z)
    # Z11: M1=1, M4=1, M5=-1, M7=1
    # Z12: M3=1, M5=1
    # Z21: M2=1, M4=1
    # Z22: M1=1, M2=-1, M3=1, M6=1
    
    w_coeffs = [
        [1, 0, 0, 1], # Z11 for M1, M2, M3, M4, M5, M6, M7? No, row is Z, col is M
        # Let's define w_coeffs[z_idx][m_idx]
        # Z11 (0): M1(1), M2(0), M3(0), M4(1), M5(-1), M6(0), M7(1)
        [1, 0, 0, 1, -1, 0, 1],
        # Z12 (1): M1(0), M2(0), M3(1), M4(0), M5(1), M6(0), M7(0)
        [0, 0, 1, 0, 1, 0, 0],
        # Z21 (2): M1(0), M2(1), M3(0), M4(1), M5(0), M6(0), M7(0)
        [0, 1, 0, 1, 0, 0, 0],
        # Z22 (3): M1(1), M2(-1), M3(1), M4(0), M5(0), M6(1), M7(0)
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Mapping for 8 sub-problems
    # s=0..3: A1 * B_top[s]
    # s=4..7: A2 * B_bot[s-4]
    
    for s in range(8):
        base_col = s * 7
        if s < 4:
            # X = A1 (cols 0,1 of A)
            # A indices: 0,1 (row 0), 4,5 (row 1)
            # X11=A[0,0] (idx 0), X12=A[0,1] (idx 1), X21=A[1,0] (idx 4), X22=A[1,1] (idx 5)
            x_indices = [0, 1, 4, 5]
            
            # Y = B_top[s] (rows 0,1 of B, cols 2s, 2s+1)
            # B indices: row k, col l -> k*8 + l
            # Y11=B[0, 2s] (idx 2s), Y12=B[0, 2s+1] (idx 2s+1)
            # Y21=B[1, 2s] (idx 8+2s), Y22=B[1, 2s+1] (idx 8+2s+1)
            y_indices = [2*s, 2*s+1, 8+2*s, 8+2*s+1]
            
            # Z = C[s] (cols 2s, 2s+1 of C)
            # C indices: col l, row i -> l*2 + i
            # Z11=C[2s, 0] (idx 2s*2+0 = 4s), Z12=C[2s+1, 0] (idx (2s+1)*2+0 = 4s+2)
            # Z21=C[2s, 1] (idx 4s+1), Z22=C[2s+1, 1] (idx 4s+3)
            z_indices = [4*s, 4*s+2, 4*s+1, 4*s+3]
            
        else:
            sp = s - 4
            # X = A2 (cols 2,3 of A)
            # X11=A[0,2] (idx 2), X12=A[0,3] (idx 3), X21=A[1,2] (idx 6), X22=A[1,3] (idx 7)
            x_indices = [2, 3, 6, 7]
            
            # Y = B_bot[sp] (rows 2,3 of B, cols 2sp, 2sp+1)
            # Y11=B[2, 2sp] (idx 16+2sp), Y12=B[2, 2sp+1] (idx 16+2sp+1)
            # Y21=B[3, 2sp] (idx 24+2sp), Y22=B[3, 2sp+1] (idx 24+2sp+1)
            y_indices = [16+2*sp, 16+2*sp+1, 24+2*sp, 24+2*sp+1]
            
            # Z = C[sp] (cols 2sp, 2sp+1 of C)
            # Same as before, z_indices for sp
            z_indices = [4*sp, 4*sp+2, 4*sp+1, 4*sp+3]
            
        for m_idx in range(7):
            col = base_col + m_idx
            # Fill U
            for x_idx, val in zip(x_indices, u_coeffs[m_idx]):
                if val != 0:
                    U[x_idx][col] += val
            # Fill V
            for y_idx, val in zip(y_indices, v_coeffs[m_idx]):
                if val != 0:
                    V[y_idx][col] += val
            # Fill W
            for z_idx, val in zip(z_indices, w_coeffs[z_idx][m_idx]):
                if val != 0:
                    W[z_idx][col] += val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
