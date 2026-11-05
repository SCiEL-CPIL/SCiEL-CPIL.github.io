import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block decomposition: 6x6 = 2x2 blocks of 3x3
    block_size = 3
    num_blocks = 2
    
    # Rank of 2x2 matrix multiplication (Strassen) is 7
    # Rank of 3x3 matrix multiplication (Schoolbook) is 27
    # Total rank = 7 * 27 = 189
    total_rank = 189
    
    # Initialize factor matrices
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    col = 0
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the steps
    # Each step has:
    # 'A': dict of (I, J) -> coeff for A[I, J] block
    # 'B': dict of (J, K) -> coeff for B[J, K] block
    # 'C': dict of (I, K) -> coeff for C[I, K] block
    
    steps = [
        {
            'A': { (0,0): 1, (1,1): 1 },
            'B': { (0,0): 1, (1,1): 1 },
            'C': { (0,0): 1, (1,1): 1 }
        },
        {
            'A': { (1,0): 1, (1,1): 1 },
            'B': { (0,0): 1 },
            'C': { (1,0): 1, (1,1): -1 }
        },
        {
            'A': { (0,0): 1 },
            'B': { (0,1): 1, (1,1): -1 },
            'C': { (0,1): 1, (1,1): 1 }
        },
        {
            'A': { (1,1): 1 },
            'B': { (1,0): 1, (0,0): -1 },
            'C': { (0,0): 1, (1,0): 1 }
        },
        {
            'A': { (0,0): 1, (0,1): 1 },
            'B': { (1,1): 1 },
            'C': { (0,0): -1, (0,1): 1 }
        },
        {
            'A': { (1,0): 1, (0,0): -1 },
            'B': { (0,0): 1, (0,1): 1 },
            'C': { (1,1): 1 }
        },
        {
            'A': { (0,1): 1, (1,1): -1 },
            'B': { (1,0): 1, (1,1): 1 },
            'C': { (0,0): 1 }
        }
    ]
    
    dim = 6
    
    for step in steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        # Iterate over local indices for the 3x3 blocks
        # u: local row of output and X
        # w: local col of output and Y
        # v: local col of X and row of Y (summed index)
        for u in range(block_size):
            for w in range(block_size):
                for v in range(block_size):
                    
                    # Update U (Left factor)
                    # X[u,v] = sum_{I,J} A_coeffs[(I,J)] * A[I*3+u, J*3+v]
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                    
                    # Update V (Right factor)
                    # Y[v,w] = sum_{J,K} B_coeffs[(J,K)] * B[J*3+v, K*3+w]
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                    
                    # Update W (Output factor)
                    # M[u,w] contributes to C blocks defined in C_coeffs
                    # C[I,K,u,w] gets contribution with weight C_coeffs[(I,K)]
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            # Task convention: W[k*n+i, r]
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                    
                    col += 1
                    
                    # Safety check (should not trigger for this small size)
                    if time.time() > start_time + 140:
                        break
        
        if time.time() > start_time + 140:
            break
    
    return {"U": U, "V": V, "W": W}
