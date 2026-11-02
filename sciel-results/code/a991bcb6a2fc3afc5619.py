import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication M_{2,2,12} can be viewed as 6 independent M_{2,2,2} blocks.
    # We use Strassen's algorithm (rank 7) for each block.
    # Total rank = 6 * 7 = 42.
    
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen's 7 multiplications for a 2x2 block.
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: linear form on A (2x2, flattened 0,1,2,3)
    # V_coeffs: linear form on B (2x2, flattened 0,1,2,3)
    # W_coeffs: linear form on C (2x2, flattened 0,1,2,3)
    
    # Note: The task uses trace convention W[k*n+i].
    # For a 2x2 block, indices are:
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[0*2+1] = W[1]
    # C[1,0] -> W[1*2+0] = W[2]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # Strassen terms:
    # 1: (A00+A11) * (B00+B11) -> C00 += 1, C11 += 1
    # 2: (A10+A11) * (B00)     -> C10 += 1, C11 -= 1
    # 3: (A00) * (B01-B11)     -> C01 += 1, C11 += 1
    # 4: (A11) * (B10-B00)     -> C00 += 1, C01 += 1
    # 5: (A00+A01) * (B11)     -> C00 -= 1, C01 += 1
    # 6: (A10-A00) * (B00+B01) -> C11 += 1
    # 7: (A01-A11) * (B10+B11) -> C00 += 1
    
    # Let's map these to indices 0..6
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_terms = [
        # term 0
        {
            'u': {0: 1, 3: 1},  # A00 + A11
            'v': {0: 1, 3: 1},  # B00 + B11
            'w': {0: 1, 3: 1}   # C00 + C11
        },
        # term 1
        {
            'u': {2: 1, 3: 1},  # A10 + A11
            'v': {0: 1},        # B00
            'w': {2: 1, 3: -1}  # C10 - C11
        },
        # term 2
        {
            'u': {0: 1},        # A00
            'v': {1: 1, 3: -1}, # B01 - B11
            'w': {1: 1, 3: 1}   # C01 + C11
        },
        # term 3
        {
            'u': {3: 1},        # A11
            'v': {2: 1, 0: -1}, # B10 - B00
            'w': {0: 1, 1: 1}   # C00 + C01
        },
        # term 4
        {
            'u': {0: 1, 1: 1},  # A00 + A01
            'v': {3: 1},        # B11
            'w': {0: -1, 1: 1}  # -C00 + C01
        },
        # term 5
        {
            'u': {2: 1, 0: -1}, # A10 - A00
            'v': {0: 1, 1: 1},  # B00 + B01
            'w': {3: 1}         # C11
        },
        # term 6
        {
            'u': {1: 1, 3: -1}, # A01 - A11
            'v': {2: 1, 3: 1},  # B10 + B11
            'w': {0: 1}         # C00
        }
    ]
    
    for block_idx in range(num_blocks):
        for term_idx in range(rank_per_block):
            col = block_idx * rank_per_block + term_idx
            term = strassen_terms[term_idx]
            
            # Fill U
            for a_idx, coeff in term['u'].items():
                U[a_idx][col] = coeff
                
            # Fill V
            # B is 2x12. Block block_idx corresponds to columns 2*block_idx and 2*block_idx+1.
            # Local B indices:
            # 0: (0, 2*block_idx)
            # 1: (0, 2*block_idx+1)
            # 2: (1, 2*block_idx)
            # 3: (1, 2*block_idx+1)
            for b_idx, coeff in term['v'].items():
                row = b_idx // 2
                local_col = b_idx % 2
                global_col = 2 * block_idx + local_col
                V[row * p + global_col][col] = coeff
                
            # Fill W
            # W is 12x2 (p x n).
            # Trace convention W[k*n + i].
            # Block block_idx corresponds to rows 2*block_idx and 2*block_idx+1.
            # Local W indices:
            # 0: (2*block_idx, 0) -> W[2*block_idx * 2 + 0]
            # 1: (2*block_idx, 1) -> W[2*block_idx * 2 + 1]
            # 2: (2*block_idx+1, 0) -> W[(2*block_idx+1) * 2 + 0]
            # 3: (2*block_idx+1, 1) -> W[(2*block_idx+1) * 2 + 1]
            for c_idx, coeff in term['w'].items():
                row = c_idx // 2
                local_col = c_idx % 2
                global_row = 2 * block_idx + local_col
                W[global_row * n + row][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
