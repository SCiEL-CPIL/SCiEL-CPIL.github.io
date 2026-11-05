import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 6x6x6 tensor decomposition.
    
    Strategy:
    - Outer: 2x2 block Strassen (rank 7)
    - Inner: 3x3 schoolbook (rank 27)
    - Total rank: 189
    
    The output W follows the convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    dim = n  # 6
    
    block_size = 3
    rank_inner = block_size ** 3  # 27
    rank_outer = 7  # Strassen rank for 2x2
    total_rank = rank_inner * rank_outer  # 189
    
    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    
    # Strassen steps for 2x2 matrix multiplication
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: {block_index: coefficient} for forming X from A blocks
    # B_coeffs: {block_index: coefficient} for forming Y from B blocks  
    # C_coeffs: {block_index: coefficient} for adding M to C blocks
    # Block indices: (0,0), (0,1), (1,0), (1,1)
    
    steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00 * (B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11 * (B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    for A_coeffs, B_coeffs, C_coeffs in steps:
        # For each local index combination in the 3x3 blocks
        for u in range(block_size):  # local row of output/X
            for w in range(block_size):  # local col of output/Y
                for v in range(block_size):  # local col of X / row of Y (summed)
                    # X[u,v] * Y[v,w] contributes to M[u,w]
                    
                    # Update U (left factor)
                    # X[u,v] = sum_{I,J} A_coeffs[(I,J)] * A[I*3+u, J*3+v]
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                    
                    # Update V (right factor)
                    # Y[v,w] = sum_{J,K} B_coeffs[(J,K)] * B[J*3+v, K*3+w]
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                    
                    # Update W (output factor)
                    # M[u,w] contributes to C blocks with weight C_coeffs[(I,K)]
                    # Task convention: W[k*n+i, r]
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
