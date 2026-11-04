import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 + Rank 16 for 4x4x1 = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Construct 4x4x4 part (cols 0-3 of B)
    # Use Kronecker product of 2x2 Strassen matrices
    
    # 2x2 Strassen U, V, W matrices (4x7)
    U22 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V22 = [
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W22 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Kronecker product to get 4x4 matrices (16x49)
    # Index mapping for 4x4:
    # A index: i*m + j -> (i1*2+i2)*2 + (j1*2+j2) = 4*i1*2 + 2*i2 + 2*j1 + j2? No.
    # Standard row-major: idx = i*4 + j.
    # i = i1*2 + i2, j = j1*2 + j2.
    # idx = (i1*2 + i2)*4 + (j1*2 + j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    # Kronecker product of U22 (4x7) and U22 (4x7) gives 16x49.
    # Row idx in 4x4 corresponds to (i1, i2, j1, j2)?
    # The tensor product of two 2x2 multiplications:
    # C = A B.
    # If we view A as block matrix of 2x2 blocks, and B as block matrix of 2x2 blocks.
    # The Kronecker product of the decompositions works if the indexing is consistent.
    # U_44[row_44, col_49] = U22[row_22_a, col_27_a] * U22[row_22_b, col_27_b]
    # Where row_44 corresponds to (i, j) and row_22_a to (i1, j1) and row_22_b to (i2, j2).
    # This requires careful index mapping.
    
    # Let's just iterate over the 49 columns and fill U, V, W
    # For each of the 49 terms (k1, k2) in the 2x2 decomposition:
    # The term is U22[i1, k1] * U22[i2, k2] for A part?
    # Actually, the rank 49 decomposition is the tensor product of two rank 7 decompositions.
    # The k-th term in the rank 49 decomposition corresponds to a pair (k1, k2) where k1, k2 in 0..6.
    # k = k1 * 7 + k2.
    
    for k1 in range(7):
        for k2 in range(7):
            k = k1 * 7 + k2
            # U part
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            # A index: i = i1*2 + i2, j = j1*2 + j2
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            a_idx = i * 4 + j
                            
                            val = U22[i1 * 2 + j1][k1] * U22[i2 * 2 + j2][k2]
                            # Wait, U22 rows are (i,j) of 2x2.
                            # For the first 2x2 block (top-left), indices are (i1, j1).
                            # For the second 2x2 block (bottom-right), indices are (i2, j2).
                            # The Kronecker product structure:
                            # U_44[a_idx, k] = U22[a_idx_1, k1] * U22[a_idx_2, k2]
                            # a_idx_1 = i1 * 2 + j1
                            # a_idx_2 = i2 * 2 + j2
                            
                            a_idx_1 = i1 * 2 + j1
                            a_idx_2 = i2 * 2 + j2
                            val = U22[a_idx_1][k1] * U22[a_idx_2][k2]
                            U[a_idx][k] = val
                            
            # V part
            for j1 in range(2):
                for j2 in range(2):
                    for k1_b in range(2): # This is the col index of B, let's call it l1, l2
                        for l2 in range(2):
                            j = j1 * 2 + j2
                            l = k1_b * 2 + l2 # B col index
                            b_idx = j * 5 + l # B is 4x5, so m=4, p=5. j in 0..3, l in 0..4.
                            # But we are only doing 4x4 part, so l in 0..3.
                            # b_idx = j * 5 + l
                            
                            b_idx_1 = j1 * 2 + k1_b
                            b_idx_2 = j2 * 2 + l2
                            val = V22[b_idx_1][k1] * V22[b_idx_2][k2]
                            V[b_idx][k] = val
                            
            # W part
            for k1_c in range(2): # C row index i, col index l
                for l1 in range(2):
                    for k2_c in range(2):
                        for l2 in range(2):
                            i = k1_c * 2 + k2_c
                            l = l1 * 2 + l2
                            c_idx = l * 4 + i # W index is k*n + i. Here l is the B col index (0..3), i is C row index.
                            # Wait, W index formula: W[k*n + i, r].
                            # k is B col index. i is C row index.
                            # So c_idx = l * 4 + i.
                            
                            c_idx_1 = k1_c * 2 + l1
                            c_idx_2 = k2_c * 2 + l2
                            val = W22[c_idx_1][k1] * W22[c_idx_2][k2]
                            W[c_idx][k] = val

    # 2. Construct 4x4x1 part (col 4 of B)
    # Rank 16
    # A is 4x4, B is 4x1, C is 4x1.
    # C_i = sum_j A_{i,j} B_{j,4}
    # Decomposition: for each i, j, one term.
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            a_idx = i * 4 + j
            b_idx = j * 5 + 4
            c_idx = 4 * 4 + i # k=4, n=4. 4*4 + i = 16 + i
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
