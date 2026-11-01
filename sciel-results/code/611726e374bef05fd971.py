import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen coefficients for 2x2 multiplication
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11 (local)
    # C indices: 0:00, 1:10, 2:01, 3:11 (local, order: C00, C10, C01, C11)
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6 (a10 - a00)
        [0, 1, 0, -1]  # M7 (a01 - a11)
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4 (b10 - b00)
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1], # M3
        [-1, 1, 0, 0], # M4
        [1, 0, 1, 0], # M5
        [0, 0, 0, 1], # M6
        [-1, 0, 0, 0]  # M7
    ]
    
    # Fill first block (cols 0,1 of B and C)
    for k in range(7):
        # U is same for both blocks
        for i in range(4):
            U[i][k] = U_strassen[k][i]
        
        # V mapping: Local B (0,1,2,3) -> Global B (0,1,4,5)
        V[0][k] = V_strassen[k][0] # b00
        V[1][k] = V_strassen[k][1] # b01
        V[4][k] = V_strassen[k][2] # b10
        V[5][k] = V_strassen[k][3] # b11
        
        # W mapping: Local C (0,1,2,3) -> Global C (0,1,2,3)
        # Global C index is k*n + i.
        # Local 0 (C00) -> Global 0 (0*2+0)
        # Local 1 (C10) -> Global 1 (0*2+1) -- Wait.
        # C[i,k]. Index k*n + i.
        # C00: k=0, i=0 -> 0.
        # C10: k=0, i=1 -> 1.
        # C01: k=1, i=0 -> 2.
        # C11: k=1, i=1 -> 3.
        # So Local order [C00, C10, C01, C11] maps to Global [0, 1, 2, 3].
        W[0][k] = W_strassen[k][0]
        W[1][k] = W_strassen[k][1]
        W[2][k] = W_strassen[k][2]
        W[3][k] = W_strassen[k][3]
        
    # Fill second block (cols 2,3 of B and C)
    for k in range(7):
        # U is same
        for i in range(4):
            U[i][k + 7] = U_strassen[k][i]
            
        # V mapping: Local B (0,1,2,3) -> Global B (2,3,6,7)
        V[2][k + 7] = V_strassen[k][0] # b02
        V[3][k + 7] = V_strassen[k][1] # b03
        V[6][k + 7] = V_strassen[k][2] # b12
        V[7][k + 7] = V_strassen[k][3] # b13
        
        # W mapping: Local C (0,1,2,3) -> Global C (4,5,6,7)
        # C02: k=2, i=0 -> 4.
        # C12: k=2, i=1 -> 5.
        # C03: k=3, i=0 -> 6.
        # C13: k=3, i=1 -> 7.
        W[4][k + 7] = W_strassen[k][0]
        W[5][k + 7] = W_strassen[k][1]
        W[6][k + 7] = W_strassen[k][2]
        W[7][k + 7] = W_strassen[k][3]
        
    return {"U": U, "V": V, "W": W}
