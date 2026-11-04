"""Matrix multiplication 6x6x6 - rank-189 Strassen-based decomposition."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 6
    
    # Use Strassen (rank 7) for 2x2 blocks, schoolbook (rank 27) for 3x3 leaves
    # Structure: 3x3 grid of 2x2 blocks, each using Strassen
    # Total rank = 27 * 7 = 189
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2 = 3
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # For matrices A, B (2x2), C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Strassen product has coefficient patterns:
    # (U_pattern, V_pattern, W_pattern) where each is dict: (row,col) -> coeff
    strassen_products = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1}),
        # M2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 
         {(1,0): 1, (1,1): -1}),
        # M3: A00(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, 
         {(0,1): 1, (1,1): 1}),
        # M4: A11(B01-B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, 
         {(0,0): 1, (1,0): 1, (1,1): 1}),
        # M5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 
         {(0,1): -1, (0,0): -1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 1}),
    ]
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col_idx = 0
    for I in range(blocks_per_dim):  # block row of A / block row of C
        for J in range(blocks_per_dim):  # block col of A / block row of B
            for K in range(blocks_per_dim):  # block col of B / block col of C
                for u_pat, v_pat, w_pat in strassen_products:
                    # Fill U: A block (I,J) with pattern u_pat
                    for (pi, pj), val in u_pat.items():
                        row = I * block_size + pi
                        col = J * block_size + pj
                        U[row * m + col][col_idx] += val
                    
                    # Fill V: B block (J,K) with pattern v_pat
                    for (pi, pj), val in v_pat.items():
                        row = J * block_size + pi
                        col = K * block_size + pj
                        V[row * p + col][col_idx] += val
                    
                    # Fill W: C block (I,K) with pattern w_pat
                    # W index convention: W[k*n + i, t]
                    for (pi, pj), val in w_pat.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
