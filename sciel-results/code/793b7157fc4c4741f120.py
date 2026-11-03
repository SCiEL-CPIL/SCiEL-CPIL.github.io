import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # We split B into B1 (4x4) and B2 (4x3)
    # Part 1: A * B1 using Strassen (Rank 7)
    # Part 2: A * B2 using Schoolbook (Rank 48)
    
    rank_total = 7 + 48
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Indices for A (4x4)
    # A[i, j] -> U[i*m + j]
    # Indices for B1 (4x4, cols 0-3 of B)
    # B[j, k] -> V[j*p + k]
    # Indices for B2 (4x3, cols 4-6 of B)
    # B[j, k] -> V[j*p + k]
    # Indices for C1 (4x4, cols 0-3 of C)
    # C[i, k] -> W[k*n + i]
    # Indices for C2 (4x3, cols 4-6 of C)
    # C[i, k] -> W[k*n + i]
    
    # Helper to get indices of a 2x2 block in a 4x4 matrix
    # Block (bi, bj) where bi, bj in {0, 1}
    # Rows: 2*bi to 2*bi+1, Cols: 2*bj to 2*bj+1
    def get_block_indices(bi, bj, dim, stride):
        # dim is the dimension of the square block (2x2)
        # stride is the row stride of the parent matrix
        # For A: dim=4, stride=4? No, A is 4x4.
        # Indices are row * 4 + col.
        # Block (bi, bj) covers rows [2*bi, 2*bi+1] and cols [2*bj, 2*bj+1]
        indices = []
        for r in range(2):
            for c in range(2):
                row = 2 * bi + r
                col = 2 * bj + c
                indices.append(row * 4 + col)
        return indices

    # Strassen Part (Rank 7)
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # A blocks:
    A11 = get_block_indices(0, 0, 2, 4)
    A12 = get_block_indices(0, 1, 2, 4)
    A21 = get_block_indices(1, 0, 2, 4)
    A22 = get_block_indices(1, 1, 2, 4)
    
    # B1 blocks (cols 0-3 of B):
    # B1 is 4x4. V index is j*7 + k.
    # Block (bi, bj) covers rows [2*bi, 2*bi+1] and cols [2*bj, 2*bj+1]
    def get_B1_block_indices(bi, bj):
        indices = []
        for r in range(2):
            for c in range(2):
                row = 2 * bi + r
                col = 2 * bj + c
                indices.append(row * 7 + col)
        return indices

    B11 = get_B1_block_indices(0, 0)
    B12 = get_B1_block_indices(0, 1)
    B21 = get_B1_block_indices(1, 0)
    B22 = get_B1_block_indices(1, 1)
    
    # C1 blocks (cols 0-3 of C):
    # C1 is 4x4. W index is k*4 + i.
    # Block (bi, bj) covers rows (cols of C) [2*bi, 2*bi+1] and cols (rows of C) [2*bj, 2*bj+1]
    # Wait, W index is k*n + i. k is row of C, i is col of C.
    # So "row" in block logic corresponds to k, "col" to i.
    def get_C1_block_indices(bi, bj):
        indices = []
        for r in range(2): # k offset
            for c in range(2): # i offset
                row = 2 * bi + r
                col = 2 * bj + c
                indices.append(row * 4 + col)
        return indices

    C11 = get_C1_block_indices(0, 0)
    C12 = get_C1_block_indices(0, 1)
    C21 = get_C1_block_indices(1, 0)
    C22 = get_C1_block_indices(1, 1)
    
    col = 0
    # P1
    for idx in A11 + A22: U[idx][col] = 1
    for idx in B11 + B22: V[idx][col] = 1
    for idx in C11: W[idx][col] = 1
    for idx in C22: W[idx][col] = 1
    col += 1
    
    # P2
    for idx in A21 + A22: U[idx][col] = 1
    for idx in B11: V[idx][col] = 1
    for idx in C21: W[idx][col] = 1
    for idx in C22: W[idx][col] = -1
    col += 1
    
    # P3
    for idx in A11: U[idx][col] = 1
    for idx in B12: V[idx][col] = 1
    for idx in B22: V[idx][col] = -1
    for idx in C12: W[idx][col] = 1
    for idx in C22: W[idx][col] = 1
    col += 1
    
    # P4
    for idx in A22: U[idx][col] = 1
    for idx in B21: V[idx][col] = 1
    for idx in B11: V[idx][col] = -1
    for idx in C11: W[idx][col] = 1
    for idx in C21: W[idx][col] = 1
    col += 1
    
    # P5
    for idx in A11 + A12: U[idx][col] = 1
    for idx in B22: V[idx][col] = 1
    for idx in C11: W[idx][col] = -1
    for idx in C12: W[idx][col] = 1
    col += 1
    
    # P6
    for idx in A21: U[idx][col] = 1
    for idx in A11: U[idx][col] = -1
    for idx in B11 + B12: V[idx][col] = 1
    for idx in C22: W[idx][col] = 1
    col += 1
    
    # P7
    for idx in A12: U[idx][col] = 1
    for idx in A22: U[idx][col] = -1
    for idx in B21 + B22: V[idx][col] = 1
    for idx in C11: W[idx][col] = 1
    col += 1
    
    # Schoolbook Part (Rank 48)
    # A (4x4) * B2 (4x3)
    # B2 cols are 4, 5, 6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
