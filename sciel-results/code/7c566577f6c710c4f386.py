import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 construction
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_strassen[a, k], V_strassen[b, k], W_strassen[c, k]
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # Term 1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # Term 2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1
    
    # Term 3
    U_strassen[0, 2] = 1
    V_strassen[2, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[1, 2] = 1; W_strassen[3, 2] = 1
    
    # Term 4
    U_strassen[3, 3] = 1
    V_strassen[1, 3] = 1; V_strassen[3, 3] = 1
    W_strassen[0, 3] = 1; W_strassen[2, 3] = 1
    
    # Term 5
    U_strassen[0, 4] = 1; U_strassen[2, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[1, 4] = 1
    
    # Term 6
    U_strassen[0, 5] = -1; U_strassen[3, 5] = 1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1
    
    # Term 7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = -1
    
    col = 0
    # 5 groups of 2 columns
    for g in range(5):
        c1 = 2 * g
        c2 = 2 * g + 1
        
        # Local B indices to Global B indices
        # Local 0: (0, c1) -> 0*p + c1
        # Local 1: (0, c2) -> 0*p + c2
        # Local 2: (1, c1) -> 1*p + c1
        # Local 3: (1, c2) -> 1*p + c2
        v_map = [c1, c2, p + c1, p + c2]
        
        # Local C indices to Global C indices
        # Local 0: (0, c1) -> c1*n + 0
        # Local 1: (0, c2) -> c2*n + 0
        # Local 2: (1, c1) -> c1*n + 1
        # Local 3: (1, c2) -> c2*n + 1
        w_map = [c1 * n, c2 * n, c1 * n + 1, c2 * n + 1]
        
        for k in range(7):
            # Fill U
            for a in range(4):
                U[a][col] = U_strassen[a, k]
            
            # Fill V
            for local_b in range(4):
                val = V_strassen[local_b, k]
                if val != 0:
                    global_b = v_map[local_b]
                    V[global_b][col] = val
            
            # Fill W
            for local_c in range(4):
                val = W_strassen[local_c, k]
                if val != 0:
                    global_c = w_map[local_c]
                    W[global_c][col] = val
            
            col += 1
    
    # 1 remaining column (index 10)
    c3 = 10
    
    # Schoolbook for 2x2x1
    # A00 * B0,c3 -> C0,c3
    U[0][col] = 1
    V[c3][col] = 1
    W[c3 * n][col] = 1
    col += 1
    
    # A01 * B1,c3 -> C0,c3
    U[1][col] = 1
    V[p + c3][col] = 1
    W[c3 * n][col] = 1
    col += 1
    
    # A10 * B0,c3 -> C1,c3
    U[2][col] = 1
    V[c3][col] = 1
    W[c3 * n + 1][col] = 1
    col += 1
    
    # A11 * B1,c3 -> C1,c3
    U[3][col] = 1
    V[p + c3][col] = 1
    W[c3 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
