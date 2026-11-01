import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition of 2x3x3
    # Rank 6
    # U_3: 6x6, V_3: 9x6, W_3: 6x6
    
    U_3 = [[0]*6 for _ in range(n*m)]
    V_3 = [[0]*6 for _ in range(m*3)] # m=3, p_block=3
    W_3 = [[0]*6 for _ in range(3*n)]
    
    # Schoolbook for 2x3x3
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U_3[i*m + j][col] = 1
                V_3[j*3 + k][col] = 1
                W_3[k*n + i][col] = 1
                col += 1
                
    rank = 12
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1: k=0,1,2
    for c in range(6):
        # U is same
        for r in range(n*m):
            U[r][c] = U_3[r][c]
            
        # V
        for r3 in range(m*3):
            if V_3[r3][c] != 0:
                j = r3 // 3
                k_local = r3 % 3
                k_global = k_local
                r_full = j * p + k_global
                V[r_full][c] = V_3[r3][c]
                
        # W
        for r3 in range(3*n):
            if W_3[r3][c] != 0:
                k_local = r3 // n
                i_local = r3 % n
                k_global = k_local
                r_full = k_global * n + i_local
                W[r_full][c] = W_3[r3][c]
                
    # Block 2: k=3,4,5
    for c in range(6, 12):
        c_local = c - 6
        # U is same
        for r in range(n*m):
            U[r][c] = U_3[r][c_local]
            
        # V
        for r3 in range(m*3):
            if V_3[r3][c_local] != 0:
                j = r3 // 3
                k_local = r3 % 3
                k_global = 3 + k_local
                r_full = j * p + k_global
                V[r_full][c] = V_3[r3][c_local]
                
        # W
        for r3 in range(3*n):
            if W_3[r3][c_local] != 0:
                k_local = r3 // n
                i_local = r3 % n
                k_global = 3 + k_local
                r_full = k_global * n + i_local
                W[r_full][c] = W_3[r3][c_local]
                
    return {"U": U, "V": V, "W": W}
