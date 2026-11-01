import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices
    # We define them as lists of lists
    
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B10)
    # P5 = (A00+A10)*B11
    # P6 = (A01-A10)*(B00+B10)
    # P7 = (A01-A11)*(B01+B11)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_block = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    V_block = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_block = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        for r in range(7):
            col = t * 7 + r
            
            # Fill U
            for i in range(n * m):
                U[i][col] = U_block[r][i]
                
            # Fill V
            # Global V index = row * p + col_idx
            # Local 0: 0, 2t -> 2t
            # Local 1: 0, 2t+1 -> 2t+1
            # Local 2: 1, 2t -> p + 2t
            # Local 3: 1, 2t+1 -> p + 2t + 1
            v_indices = [2*t, 2*t+1, p+2*t, p+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_block[r][l]
                
            # Fill W
            # Convention W[k*n + i, r]
            # Local 0: c[0][2t] -> i=0, k=2t -> 2t*n
            # Local 1: c[0][2t+1] -> i=0, k=2t+1 -> (2t+1)*n
            # Local 2: c[1][2t] -> i=1, k=2t -> 2t*n + 1
            # Local 3: c[1][2t+1] -> i=1, k=2t+1 -> (2t+1)*n + 1
            w_indices = [2*t*n, (2*t+1)*n, 2*t*n + 1, (2*t+1)*n + 1]
            for l in range(4):
                W[w_indices[l]][col] = W_block[r][l]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
