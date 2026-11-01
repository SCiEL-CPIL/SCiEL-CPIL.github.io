"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 30
    P = p * n  # 30
    
    # We use a block decomposition.
    # 15 columns of B are split into 7 blocks of 2 columns and 1 block of 1 column.
    # Each 2-column block uses Strassen's algorithm (rank 7).
    # The 1-column block uses schoolbook (rank 4).
    # Total rank = 7 * 7 + 4 = 53.
    
    rank = 53
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Define Strassen coefficients
    # U[r][i, j]
    # V[r][j, k]
    # W[r][k, i]
    
    strassen_U = [
        [[1, 0], [0, 1]],  # P1: A00, A11
        [[0, 0], [1, 1]],  # P2: A10, A11
        [[1, 0], [0, 0]],  # P3: A00
        [[0, 0], [0, 1]],  # P4: A11
        [[1, 1], [0, 0]],  # P5: A00, A01
        [[-1, 0], [1, 0]], # P6: -A00, A10
        [[0, 1], [0, -1]], # P7: A01, -A11
    ]
    
    strassen_V = [
        [[1, 1], [0, 0]],  # P1: B00, B11
        [[1, 0], [0, 0]],  # P2: B00
        [[0, 1], [0, -1]], # P3: B01, -B11
        [[0, -1], [1, 0]], # P4: -B01, B10
        [[0, 0], [0, 1]],  # P5: B11
        [[1, 1], [0, 0]],  # P6: B00, B01
        [[0, 0], [1, 1]],  # P7: B10, B11
    ]
    
    # W coefficients for C
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # W[r][k, i]
    strassen_W = [
        [[1, 0], [0, 1]],  # P1: C00, C11
        [[0, 1], [-1, 0]], # P2: C10, -C11
        [[0, 0], [1, 1]],  # P3: C01, C11
        [[1, 1], [0, 0]],  # P4: C00, C10
        [[-1, 0], [1, 0]], # P5: -C00, C01
        [[0, 0], [0, 1]],  # P6: C11
        [[1, 0], [0, 0]],  # P7: C00
    ]
    
    # Process 7 blocks of 2 columns
    for block_idx in range(7):
        k_start = block_idx * 2
        # For each Strassen term r
        for r in range(7):
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = strassen_U[r][i][j]
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = k_start + k_local
                    V[j * p + k_global, col] = strassen_V[r][j][k_local]
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    k_global = k_start + k_local
                    W[k_global * n + i, col] = strassen_W[r][k_local][i]
            
            col += 1
    
    # Process 1 block of 1 column (column 14)
    k_start = 14
    # Schoolbook 2x2x1
    # P1 = A00 B00 -> C00
    # P2 = A01 B10 -> C00
    # P3 = A10 B00 -> C10
    # P4 = A11 B10 -> C10
    
    # Term 1
    U[0 * m + 0, col] = 1 # A00
    V[0 * p + k_start, col] = 1 # B00
    W[k_start * n + 0, col] = 1 # C00
    col += 1
    
    # Term 2
    U[0 * m + 1, col] = 1 # A01
    V[1 * p + k_start, col] = 1 # B10
    W[k_start * n + 0, col] = 1 # C00
    col += 1
    
    # Term 3
    U[1 * m + 0, col] = 1 # A10
    V[0 * p + k_start, col] = 1 # B00
    W[k_start * n + 1, col] = 1 # C10
    col += 1
    
    # Term 4
    U[1 * m + 1, col] = 1 # A11
    V[1 * p + k_start, col] = 1 # B10
    W[k_start * n + 1, col] = 1 # C10
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
