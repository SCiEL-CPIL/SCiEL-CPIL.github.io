import time
import sys

def construct():
    n, m, p = 2, 2, 13
    rank = 6 * 7 + 4  # 42 + 4 = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U: coeffs of A_ij in M_r
    # V: coeffs of B_jk in M_r
    # W: coeffs of C_ik in M_r (Note: W index is k*n+i)
    
    # r=1..7
    # A indices: 00, 01, 10, 11 -> U rows 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> V rows 0, 1, 2, 3 (within block)
    # C indices: 00, 01, 10, 11 -> W rows 0, 1, 2, 3 (within block, k*n+i)
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W_strassen = [
        # C00: M1 + M4 - M5 + M7
        [1, 0, 0, 1, -1, 0, 1],
        # C01: M3 + M5
        [0, 0, 1, 0, 1, 0, 0],
        # C10: M2 + M4
        [0, 1, 0, 1, 0, 0, 0],
        # C11: M1 - M2 + M3 + M6
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col_offset = 0
    for b in range(6):
        for r in range(7):
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col_offset + r] = U_strassen[i * m + j][r]
            
            # Fill V
            for j in range(m):
                for k in range(2):
                    global_k = b * 2 + k
                    V[j * p + global_k][col_offset + r] = V_strassen[j * 2 + k][r]
            
            # Fill W
            for k in range(2):
                global_k = b * 2 + k
                for i in range(n):
                    W[global_k * n + i][col_offset + r] = W_strassen[k * n + i][r]
        
        col_offset += 7
    
    # Last column (k=12) using schoolbook
    # Rank 4
    # Terms: A00*B00->C00, A01*B10->C00, A10*B00->C10, A11*B10->C11
    # Wait, schoolbook for 2x2x1:
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    
    # r=0: A00, B00, C00
    U[0][col_offset + 0] = 1
    V[0 * p + 12][col_offset + 0] = 1
    W[12 * n + 0][col_offset + 0] = 1
    
    # r=1: A01, B10, C00
    U[1][col_offset + 1] = 1
    V[1 * p + 12][col_offset + 1] = 1
    W[12 * n + 0][col_offset + 1] = 1
    
    # r=2: A10, B00, C10
    U[2][col_offset + 2] = 1
    V[0 * p + 12][col_offset + 2] = 1
    W[12 * n + 1][col_offset + 2] = 1
    
    # r=3: A11, B10, C11
    U[3][col_offset + 3] = 1
    V[1 * p + 12][col_offset + 3] = 1
    W[12 * n + 1][col_offset + 3] = 1
    
    return {"U": U, "V": V, "W": W}
