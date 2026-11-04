import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 32
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (col 0) * B_1 (row 0)
    # A_1 is 2x1, B_1 is 1x6.
    # U indices for A_1: 0 (row 0), 3 (row 1)
    # V indices for B_1: 0..5 (row 0)
    # W indices for C_1: C_{r,k} -> W[2k+r]
    
    for k in range(p):
        # Mult 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # Mult 2: A[1,0] * B[0,k] -> C[1,k]
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
    # Part 2: A_2 (cols 1,2) * B_2 (rows 1,2)
    # A_2 is 2x2, B_2 is 2x6.
    # Split B_2 into 3 blocks of 2x2.
    # A_2 indices in U:
    # a00 = A[0,1] -> U[1]
    # a01 = A[0,2] -> U[2]
    # a10 = A[1,1] -> U[4]
    # a11 = A[1,2] -> U[5]
    
    for x in range(3):
        c_start = 2 * x
        
        # B_2x indices in V:
        # b00 = B[1, c_start]     -> V[6 + c_start]
        # b01 = B[1, c_start + 1] -> V[6 + c_start + 1]
        # b10 = B[2, c_start]     -> V[12 + c_start]
        # b11 = B[2, c_start + 1] -> V[12 + c_start + 1]
        
        # W indices for C_x:
        # C_x[0,0] -> C[0, c_start]     -> W[2*c_start]
        # C_x[0,1] -> C[0, c_start+1]   -> W[2*c_start + 2]
        # C_x[1,0] -> C[1, c_start]     -> W[2*c_start + 1]
        # C_x[1,1] -> C[1, c_start+1]   -> W[2*c_start + 3]
        
        # Strassen M1: (a00+a11) * (b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1; V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # Strassen M2: (a10+a11) * b00
        U[4][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1
        W[2*c_start + 1][col] = 1
        W[2*c_start + 3][col] = -1
        col += 1
        
        # Strassen M3: a00 * (b01-b11)
        U[1][col] = 1
        V[6 + c_start + 1][col] = 1
        V[12 + c_start + 1][col] = -1
        W[2*c_start + 2][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # Strassen M4: a11 * (b10-b00)
        U[5][col] = 1
        V[12 + c_start][col] = 1
        V[6 + c_start][col] = -1
        W[2*c_start][col] = 1
        W[2*c_start + 1][col] = 1
        col += 1
        
        # Strassen M5: (a00+a01) * b11
        U[1][col] = 1; U[2][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = -1
        W[2*c_start + 2][col] = 1
        col += 1
        
        # Strassen M6: (a10-a00) * (b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[6 + c_start][col] = 1
        V[6 + c_start + 1][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # Strassen M7: (a01-a11) * (b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[12 + c_start][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        col += 1
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_est = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
                
    if np.max(np.abs(T_est - T_target)) > 1e-9:
        raise ValueError("Decomposition verification failed")
        
    return {"U": U, "V": V, "W": W}
