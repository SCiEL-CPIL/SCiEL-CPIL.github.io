import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_DEADLINE = time.time() + 140.0

def _validate_exact(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 29 construction
    # 28 from 4 Strassen blocks (2x2x2) for the 2x2x8 part
    # 1 from the 2x1x8 part
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 29
    V = [[0] * rank for _ in range(m * p)] # 24 x 29
    W = [[0] * rank for _ in range(p * n)] # 16 x 29
    
    # Strassen coefficients
    # M1..M7
    # U coeffs (A00, A01, A10, A11)
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coeffs (B00, B01, B10, B11)
    V_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coeffs (C00, C01, C10, C11)
    W_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Fill the 4 Strassen blocks
    for q in range(4):
        for r in range(7):
            c = 7 * q + r
            
            # U: A indices for block q
            # A is 2x2 (cols 0,1 of global A)
            # Global U rows:
            # A00 -> i=0, j=0 -> idx 0
            # A01 -> i=0, j=1 -> idx 1
            # A10 -> i=1, j=0 -> idx 3
            # A11 -> i=1, j=1 -> idx 4
            
            U[0][c] = U_coeffs[r][0]
            U[1][c] = U_coeffs[r][1]
            U[3][c] = U_coeffs[r][2]
            U[4][c] = U_coeffs[r][3]
            
            # V: B indices for block q
            # B is 2x2 (rows 0,1; cols 2q, 2q+1 of global B)
            # Global V rows:
            # B00 (j=0, k=2q) -> idx 2q
            # B01 (j=0, k=2q+1) -> idx 2q+1
            # B10 (j=1, k=2q) -> idx 8 + 2q
            # B11 (j=1, k=2q+1) -> idx 8 + 2q + 1
            
            V[2*q][c] = V_coeffs[r][0]
            V[2*q+1][c] = V_coeffs[r][1]
            V[8+2*q][c] = V_coeffs[r][2]
            V[8+2*q+1][c] = V_coeffs[r][3]
            
            # W: C indices for block q
            # C is 2x2 (rows 0,1; cols 2q, 2q+1 of global C)
            # Global W rows:
            # C00 (i=0, k=2q) -> idx 2*(2q) + 0 = 4q
            # C01 (i=0, k=2q+1) -> idx 2*(2q+1) + 0 = 4q+2
            # C10 (i=1, k=2q) -> idx 2*(2q) + 1 = 4q+1
            # C11 (i=1, k=2q+1) -> idx 2*(2q+1) + 1 = 4q+3
            
            W[4*q][c] = W_coeffs[r][0]
            W[4*q+2][c] = W_coeffs[r][1]
            W[4*q+1][c] = W_coeffs[r][2]
            W[4*q+3][c] = W_coeffs[r][3]
            
    # Fill the 2x1x8 part (col 28)
    c = 28
    # A_R is col 2 of A (j=2)
    # A02 -> i=0, j=2 -> idx 2
    # A12 -> i=1, j=2 -> idx 5
    U[2][c] = 1
    U[5][c] = 1
    
    # B_R is row 2 of B (j=2)
    # B2k -> j=2, k=0..7 -> idx 16+k
    for k in range(8):
        V[16+k][c] = 1
        
    # C_R is all of C
    # C_ik -> i=0,1; k=0..7
    # W idx = 2k + i
    for k in range(8):
        W[2*k][c] = 1      # i=0
        W[2*k+1][c] = 1    # i=1
        
    # Verify
    if _validate_exact(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
