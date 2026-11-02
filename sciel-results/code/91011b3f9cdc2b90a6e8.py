import time
import numpy as np

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_{2,2,2}
    # A vec: [A00, A01, A10, A11]
    # B vec: [B00, B01, B10, B11]
    # C vec: [C00, C01, C10, C11]
    
    A_s = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    B_s = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    C_s = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    # Schoolbook coefficients for M_{2,2,1}
    # A vec: [A00, A01, A10, A11]
    # B vec: [B0, B1] (for the 2x1 block)
    # C vec: [C0, C1] (for the 2x1 block)
    
    A_b = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    B_b = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    C_b = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    col_offset = 0
    
    # 7 Strassen blocks for k = 0..13
    for block in range(7):
        k0 = block * 2
        k1 = k0 + 1
        for r in range(7):
            c = col_offset + r
            # U
            for i in range(4):
                U[i][c] = A_s[r][i]
            # V
            # V row index: j*p + k
            for j in range(2):
                V[j*p + k0][c] = B_s[r][j*2 + 0]
                V[j*p + k1][c] = B_s[r][j*2 + 1]
            # W
            # W row index: k*n + i
            # C_s[r] is [C00, C01, C10, C11]
            # C00 -> k0, i=0
            # C01 -> k1, i=0
            # C10 -> k0, i=1
            # C11 -> k1, i=1
            W[k0*n + 0][c] = C_s[r][0]
            W[k1*n + 0][c] = C_s[r][1]
            W[k0*n + 1][c] = C_s[r][2]
            W[k1*n + 1][c] = C_s[r][3]
        col_offset += 7
        
    # 1 schoolbook block for k = 14
    k_last = 14
    for r in range(4):
        c = col_offset + r
        # U
        for i in range(4):
            U[i][c] = A_b[r][i]
        # V
        V[0*p + k_last][c] = B_b[r][0]
        V[1*p + k_last][c] = B_b[r][1]
        # W
        W[k_last*n + 0][c] = C_b[r][0]
        W[k_last*n + 1][c] = C_b[r][1]
        
    return {"U": U, "V": V, "W": W}
