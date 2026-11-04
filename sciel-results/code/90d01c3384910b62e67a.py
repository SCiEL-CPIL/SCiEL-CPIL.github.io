"""Strassen-based decomposition for 6x6 matrix multiplication using 2x2 block structure."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's algorithm on 2x2 blocks of 3x3 matrices
    # Each block multiplication uses schoolbook (rank 9)
    # Total rank: 7 * 9 = 63... wait, each Strassen term is a 3x3 matrix product
    # A 3x3 matrix product has rank 27 (schoolbook)
    # So total rank: 7 * 27 = 189
    
    n, m, p = DIMENSIONS  # all 6
    
    # 2x2 block structure: blocks are 3x3
    block_size = 3
    # Number of Strassen terms
    num_terms = 7
    
    # Each Strassen term requires a 3x3 x 3x3 matrix multiplication
    # Schoolbook rank for 3x3 is 9
    rank_per_term = block_size * block_size * block_size  # 27
    total_rank = num_terms * rank_per_term  # 189
    
    # Initialize factor matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define the linear combinations for each Strassen term
    # For each term, define:
    # - U coefficients: how to combine blocks of A
    # - V coefficients: how to combine blocks of B
    # - W coefficients: how to combine the result into C
    
    # Block indices: 00, 01, 10, 11
    # A block ij is A[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
    
    # Strassen terms definition:
    # term: (U_blocks, V_blocks, W_blocks)
    # U_blocks: list of (block_idx, coeff) for A
    # V_blocks: list of (block_idx, coeff) for B
    # W_blocks: list of (block_idx, coeff) for C (reconstruction)
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (2, -1), (3, 1)]},
        # M2 = (A10 + A11)B00
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3 = A00(B10 - B11)
        {'U': [(0, 1)], 'V': [(2, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        # M4 = A11(B00 - B01)
        {'U': [(3, 1)], 'V': [(0, 1), (1, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (A00 + A01)B11
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        # M6 = (A10 - A00)(B00 + B01)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (A01 - A11)(B10 + B11)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]
    
    # Block index to (row_offset, col_offset) in 6x6 matrix
    # Block ij is at rows i*3 to (i+1)*3-1, cols j*3 to (j+1)*3-1
    def block_to_indices(block_idx):
        i = block_idx // 2
        j = block_idx % 2
        return (i * block_size, j * block_size)
    
    # Build the decomposition
    col = 0
    for term_idx in range(num_terms):
        term = strassen_terms[term_idx]
        
        # For each position in the 3x3 block, create a rank-1 factor
        for a in range(block_size):  # row within block
            for c in range(block_size):  # col within block (for A)
                for b in range(block_size):  # col within block (for B)
                    if col >= total_rank:
                        break
                    
                    # U vector: A contributions
                    for block_idx, coeff in term['U']:
                        ri, cj = block_to_indices(block_idx)
                        row_idx = ri + a
                        col_idx_U = cj + c
                        U[row_idx * m + col_idx_U][col] += coeff
                    
                    # V vector: B contributions
                    for block_idx, coeff in term['V']:
                        ri, cj = block_to_indices(block_idx)
                        row_idx_V = ri + c  # c is the column of A, which is row of B
                        col_idx = cj + b
                        V[row_idx_V * p + col_idx][col] += coeff
                    
                    # W vector: C contributions
                    for block_idx, coeff in term['W']:
                        ri, cj = block_to_indices(block_idx)
                        row_idx_W = ri + a
                        col_idx_W = cj + b
                        # W index: k*n + i, where k is column of C, i is row of C
                        # In our convention: W[k*n + i, r]
                        W[row_idx_W * n + col_idx_W][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
