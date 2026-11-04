import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: 6x6 = 2x2 blocks of 3x3 matrices
    # Outer: Strassen (rank 7), Inner: Schoolbook (rank 27)
    # Total rank = 7 * 27 = 189
    
    block_size = 3
    dim = n  # 6
    
    rank_outer = 7
    rank_inner = block_size ** 3  # 27
    total_rank = rank_outer * rank_inner  # 189
    
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    # Strassen steps for 2x2 block multiplication
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}}
    ]
    
    col = 0
    
    for step in strassen_steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        # Schoolbook: iterate over all (u, v, w) in the 3x3 block
        for u in range(block_size):
            for v in range(block_size):
                for w in range(block_size):
                    # Update U
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                    
                    # Update V
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                    
                    # Update W (convention: W[k*n+i, r])
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
