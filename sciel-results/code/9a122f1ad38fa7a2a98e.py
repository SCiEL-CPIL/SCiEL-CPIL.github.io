import time
import sys

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # X indices: 0: x11, 1: x12, 2: x21, 3: x22
    # Y indices: 0: y11, 1: y12, 2: y21, 3: y22
    # Z indices: 0: z11, 1: z12, 2: z21, 3: z22
    
    # U coefficients for X
    # M1: x11 + x22
    # M2: x21 + x22
    # M3: x11
    # M4: x22
    # M5: x11 + x12
    # M6: x21 - x11
    # M7: x12 - x22
    
    U_strassen = [
        [1, 0, 0, 1, 1, -1, 0],  # x11
        [0, 1, 0, 0, 1, 0, 1],   # x12
        [0, 0, 1, 0, 0, 1, 0],   # x21
        [1, 0, 1, 1, 0, 0, -1]   # x22
    ]
    
    # V coefficients for Y
    # M1: y11 + y22
    # M2: y11
    # M3: y12 - y22
    # M4: y21 - y11
    # M5: y22
    # M6: y11 + y12
    # M7: y21 + y22
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # y11
        [0, 0, 1, 0, 0, 1, 0],   # y12
        [0, 0, 0, 1, 0, 0, 1],   # y21
        [1, 0, -1, 0, 1, 0, 1]   # y22
    ]
    
    # W coefficients for Z
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # z11
        [0, 0, 1, 0, 1, 0, 0],   # z12
        [0, 1, 0, 1, 0, 0, 0],   # z21
        [1, -1, 1, 0, 0, 1, 0]   # z22
    ]
    
    col = 0
    for k in range(4):
        # Process A_L * B_top_k
        # A_L indices in U: 
        # x11 -> A[0,0] -> 0
        # x12 -> A[0,1] -> 1
        # x21 -> A[1,0] -> 4
        # x22 -> A[1,1] -> 5
        u_idx_L = [0, 1, 4, 5]
        
        # B_top_k indices in V:
        # y11 -> B[0, 2k] -> 2k
        # y12 -> B[0, 2k+1] -> 2k+1
        # y21 -> B[1, 2k] -> 8+2k
        # y22 -> B[1, 2k+1] -> 9+2k
        v_idx_L = [2*k, 2*k+1, 8+2*k, 9+2*k]
        
        # C_k indices in W:
        # z11 -> C[0, 2k] -> 4k
        # z12 -> C[0, 2k+1] -> 4k+1
        # z21 -> C[1, 2k] -> 4k+2
        # z22 -> C[1, 2k+1] -> 4k+3
        w_idx = [4*k, 4*k+1, 4*k+2, 4*k+3]
        
        for s in range(7):
            # Fill U
            for i in range(4):
                U[u_idx_L[i]][col] = U_strassen[i][s]
            # Fill V
            for i in range(4):
                V[v_idx_L[i]][col] = V_strassen[i][s]
            # Fill W
            for i in range(4):
                W[w_idx[i]][col] = W_strassen[i][s]
            col += 1
            
        # Process A_R * B_bot_k
        # A_R indices in U:
        # x11 -> A[0,2] -> 2
        # x12 -> A[0,3] -> 3
        # x21 -> A[1,2] -> 6
        # x22 -> A[1,3] -> 7
        u_idx_R = [2, 3, 6, 7]
        
        # B_bot_k indices in V:
        # y11 -> B[2, 2k] -> 16+2k
        # y12 -> B[2, 2k+1] -> 17+2k
        # y21 -> B[3, 2k] -> 24+2k
        # y22 -> B[3, 2k+1] -> 25+2k
        v_idx_R = [16+2*k, 17+2*k, 24+2*k, 25+2*k]
        
        # W indices are the same, we add to them
        for s in range(7):
            # Fill U
            for i in range(4):
                U[u_idx_R[i]][col] = U_strassen[i][s]
            # Fill V
            for i in range(4):
                V[v_idx_R[i]][col] = V_strassen[i][s]
            # Fill W (Add to existing values)
            for i in range(4):
                W[w_idx[i]][col] = W_strassen[i][s]
            col += 1

    return {"U": U, "V": V, "W": W}
