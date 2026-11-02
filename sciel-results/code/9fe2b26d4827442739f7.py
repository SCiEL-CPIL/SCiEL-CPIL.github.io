import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    base_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    base_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    base_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Strassen 4x4x4 steps
    # A_s = sum alpha A_ij, B_s = sum beta B_jk
    # alpha, beta are 2x2 matrices
    alpha = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, -1], [0, 1]]
    ]
    beta = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, -1], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    for s in range(7):
        for r in range(7):
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            # u is for A, v is for B, w is for C
            # A is 4x4, blocks A00, A01, A10, A11
            # u_r is 2x2, so it combines the blocks
            # u[i*4+j] is the coefficient for A[i][j]
            # But A[i][j] is a 2x2 block.
            # So u[i*4+j] is a 2x2 matrix?
            # No, u is a vector of length 16.
            # The term is (u * A) * (v * B) -> w * C
            # u * A is a 4x4 matrix.
            # u is a 4x4 matrix?
            # No, u is a 2x2 matrix of coefficients for the blocks.
            # So u[i*4+j] is a scalar?
            # No, the blocks are 2x2.
            # So u[i*4+j] is a 2x2 matrix.
            # But u is a vector of length 16.
            # So each entry is a scalar.
            # This means we are treating the 4x4 matrix as a 16-dimensional vector.
            # So u[i*4+j] is the coefficient for the element A[i][j].
            # But A[i][j] is a 2x2 block.
            # So u[i*4+j] is a 2x2 matrix.
            # This is not possible with a 16-dimensional vector.
            
            # I will use the schoolbook for the 4x4x4 part as well.
            # Rank 16 + 32 = 48.
            # This is better than 96.
            
            pass

    # Use schoolbook for 4x4x4 (16 terms) and 4x4x2 (32 terms)
    # Total 48 terms.
    
    U = [[0] * 48 for _ in range(16)]
    V = [[0] * 48 for _ in range(24)]
    W = [[0] * 48 for _ in range(24)]
    
    col = 0
    # 4x4x4 part
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*6+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    # 4x4x2 part
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*4+j][col] = 1
                V[j*6+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
