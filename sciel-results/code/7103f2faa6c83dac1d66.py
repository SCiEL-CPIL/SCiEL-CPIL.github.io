"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Rank-49 decomposition of 4x4x4 matrix multiplication tensor
    # Based on the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrices
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Define the rank-7 Strassen factors for 2x2 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U_7: 4x7
    U_7 = np.zeros((4, 7), dtype=int)
    U_7[0, 0] = 1; U_7[3, 0] = 1
    U_7[2, 1] = 1; U_7[3, 1] = 1
    U_7[0, 2] = 1
    U_7[3, 3] = 1
    U_7[0, 4] = 1; U_7[1, 4] = 1
    U_7[2, 5] = 1; U_7[0, 5] = -1
    U_7[1, 6] = 1; U_7[3, 6] = -1
    
    # V_7: 4x7
    V_7 = np.zeros((4, 7), dtype=int)
    V_7[0, 0] = 1; V_7[3, 0] = 1
    V_7[0, 1] = 1
    V_7[1, 2] = 1; V_7[3, 2] = -1
    V_7[2, 3] = 1; V_7[0, 3] = -1
    V_7[3, 4] = 1
    V_7[0, 5] = 1; V_7[1, 5] = 1
    V_7[2, 6] = 1; V_7[3, 6] = 1
    
    # W_7: 4x7
    W_7 = np.zeros((4, 7), dtype=int)
    W_7[0, 0] = 1; W_7[0, 3] = 1; W_7[0, 4] = -1; W_7[0, 6] = 1
    W_7[1, 2] = 1; W_7[1, 4] = 1
    W_7[2, 1] = 1; W_7[2, 3] = 1
    W_7[3, 0] = 1; W_7[3, 1] = -1; W_7[3, 2] = 1; W_7[3, 5] = 1
    
    # Construct 16x49 matrices
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        for j in range(4):
            # A index
            idx_A = 4 * i + j
            b_A = 2 * (i // 2) + (j // 2)
            e_A = 2 * (i % 2) + (j % 2)
            
            for c1 in range(7):
                for c2 in range(7):
                    col = 7 * c1 + c2
                    U[idx_A, col] = U_7[b_A, c1] * U_7[e_A, c2]
                    
                    # B index (j, k) - wait, V depends on j, k
                    # We need to fill V and W based on their own indices
                    # This loop structure is inefficient for V and W if done independently
                    # But we can just compute them separately or in the same loop if we iterate over all indices
    
    # Better to compute V and W in separate loops or a unified one
    # Let's redo the filling for V and W correctly
    
    for j in range(4):
        for k in range(4):
            idx_B = 4 * j + k
            b_B = 2 * (j // 2) + (k // 2)
            e_B = 2 * (j % 2) + (k % 2)
            
            for c1 in range(7):
                for c2 in range(7):
                    col = 7 * c1 + c2
                    V[idx_B, col] = V_7[b_B, c1] * V_7[e_B, c2]
                    
    for k in range(4):
        for i in range(4):
            idx_C = 4 * k + i
            b_C = 2 * (k // 2) + (i // 2)
            e_C = 2 * (k % 2) + (i % 2)
            
            for c1 in range(7):
                for c2 in range(7):
                    col = 7 * c1 + c2
                    W[idx_C, col] = W_7[b_C, c1] * W_7[e_C, c2]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
