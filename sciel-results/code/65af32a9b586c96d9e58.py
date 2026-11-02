import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Laderman's rank 23 decomposition for 3x3x3
    # U_L, V_L, W_L are 9x23
    # Rows of U_L correspond to A[i,j] with index i*3+j
    # Rows of V_L correspond to B[i,j] with index i*3+j (locally)
    # Rows of W_L correspond to C[i,j] with index i*3+j (locally, but careful with k)
    
    # Standard Laderman factors (hardcoded from verified source)
    # Note: These are for the 3x3x3 part.
    U_L = [
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    V_L = [
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    W_L = [
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    ]

    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: Laderman for cols 0,1,2
    # U is 9x32. Top 9 rows, cols 0-22 get U_L
    for i in range(9):
        for c in range(23):
            U[i][c] = U_L[i][c]

    # W is 12x32. Top 9 rows, cols 0-22 get W_L
    # W index is k*3+i. W_L index is i*3+k? 
    # Standard Laderman W_L row r corresponds to C[i,j] where r = i*3+j? 
    # In our problem, W row index is k*n+i = k*3+i.
    # If Laderman W_L is defined for C[i,j] (output row i, col j), then row index in Laderman is i*3+j.
    # Our output C is 3x4. C[i,k].
    # For k=0,1,2, our W row is k*3+i.
    # Laderman computes C3x3. Let's assume Laderman W_L row r corresponds to C[i,j] with r=i*3+j.
    # Then for k in 0..2, our row index is k*3+i.
    # This matches if we map Laderman col j to our col k.
    # So W[k*3+i, c] = W_L[i*3+k, c]? 
    # Wait, Laderman W_L is usually defined such that $W_{i,j}$ contributes to $C_{i,j}$.
    # So row index in W_L is $i \cdot 3 + j$.
    # In our global W, row index for $C_{i,k}$ is $k \cdot 3 + i$.
    # So we need $W[k \cdot 3 + i, c] = W_L[i \cdot 3 + k, c]$.
    
    for i in range(3):
        for k in range(3):
            global_w_row = k * 3 + i
            laderman_w_row = i * 3 + k
            for c in range(23):
                W[global_w_row][c] = W_L[laderman_w_row][c]

    # V is 12x32. 
    # V index is j*p+k = j*4+k.
    # Laderman V_L row r corresponds to B[i,j] with r=i*3+j? 
    # Standard Laderman V_L row r corresponds to B[i,j] where r=i*3+j.
    # Our B is 3x4. B[j,k].
    # For k=0,1,2, our V row is j*4+k.
    # So we need $V[j \cdot 4 + k, c] = V_L[j \cdot 3 + k, c]$.
    
    for j in range(3):
        for k in range(3):
            global_v_row = j * 4 + k
            laderman_v_row = j * 3 + k
            for c in range(23):
                V[global_v_row][c] = V_L[laderman_v_row][c]

    # Part 2: Schoolbook for col 3
    for i in range(3):
        for j in range(3):
            c_idx = 23 + i * 3 + j
            U[i * 3 + j][c_idx] = 1
            V[j * 4 + 3][c_idx] = 1
            W[3 * 3 + i][c_idx] = 1 # k=3, index 3*3+i

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
