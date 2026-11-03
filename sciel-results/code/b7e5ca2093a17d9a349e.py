"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank-20 tensor decomposition (e.g., a known or constructed rank-20 for 6x6x6)
    # Since the schoolbook is rank 216, and Strassen gives rank 216 for 2^k sizes,
    # but for 6x6x6, there is a rank-20 decomposition (e.g., from work on tensor rank).
    # Actually, the rank of the 6x6x6 matrix multiplication tensor is not known to be 20.
    # Let me use a different approach: use a known decomposition or fall back to schoolbook.
    
    # Actually, let me just return the schoolbook decomposition as it's guaranteed to be correct.
    # The task is to find a rank-minimal exact decomposition, and the score is the rank.
    # The seed has rank 216, so we need to do better.
    
    # Let me try a Strassen-based approach for 6x6x6.
    # 6 = 3 * 2, so we can think of 6x6 matrices as 2x2 blocks of 3x3 matrices.
    # Using Strassen's algorithm on the 2x2 block structure with 3x3 entries:
    # The 3x3 matrix multiplication has rank 27 (schoolbook).
    # Strassen gives rank 7 for 2x2 matrices, but here the entries are 3x3 matrices.
    # So the total rank would be 7 * 27 = 189.
    # That's better than 216.
    
    # Actually, let me think more carefully. The matrix multiplication tensor for n x n x n has rank at most n^3.
    # For 6x6x6, schoolbook is 216.
    # Using Strassen recursively: 6 = 2 * 3, so we can use Strassen on the 2x2 block structure.
    # The base case is 3x3x3 which has rank 27.
    # Strassen's algorithm: 7 multiplications of smaller matrices.
    # So the rank would be 7 * 27 = 189.
    
    # Let me implement this Strassen-based decomposition.
    
    # The 6x6 matrix A can be written as a 2x2 block matrix where each block is 3x3:
    # A = [[A11, A12], [A21, A22]]
    # Similarly for B and C.
    
    # Strassen's algorithm computes C = A*B using 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # And then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each "product" M_t is a 3x3 matrix product.
    # The tensor rank of a 3x3x3 matrix multiplication is 27 (schoolbook).
    # So the total rank is 7 * 27 = 189.
    
    # Let me construct the decomposition.
    # The indices for A are (i, j) where i, j in {0, 1, 2, 3, 4, 5}.
    # i = 3 * block_i + sub_i, where block_i in {0, 1} and sub_i in {0, 1, 2}.
    # Similarly for j, k.
    
    # The tensor T[a, b, c] where a = (i, j) for A, b = (j', k) for B, c = (k', i') for C.
    # In the W convention, W[k*n + i, r] so c = k*n + i.
    
    # For the Strassen decomposition, each of the 7 terms corresponds to a 3x3x3 tensor product.
    # For term t, the A-part is a linear combination of blocks of A, and similarly for B and C.
    
    # Let me define the Strassen terms.
    # Each term t has:
    # - A coefficients: for each block (bi, bj) in {0,1}^2, a coefficient alpha_t[bi][bj]
    # - B coefficients: for each block (bi, bj) in {0,1}^2, a coefficient beta_t[bi][bj]
    # - C coefficients: for each block (ci, cj) in {0,1}^2, a coefficient gamma_t[ci][cj]
    
    # The 7 Strassen terms:
    strassen_terms = []
    
    # M1 = (A11 + A22) * (B11 + B22)
    # C11 += M1, C22 += M1
    strassen_terms.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A21 + A22) * B11
    # C21 += M2, C22 -= M2
    strassen_terms.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = A11 * (B12 - B22)
    # C12 += M3, C22 += M3
    strassen_terms.append({
        'A': {(0,0): 1},
        'B': {(0,1): 1, (1,1): -1},
        'C': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A22 * (B21 - B11)
    # C11 += M4, C21 += M4
    strassen_terms.append({
        'A': {(1,1): 1},
        'B': {(1,0): 1, (0,0): -1},
        'C': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A11 + A12) * B22
    # C11 -= M5, C12 += M5
    strassen_terms.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (A21 - A11) * (B11 + B12)
    # C22 += M6
    strassen_terms.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,1): 1}
    })
    
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 += M7
    strassen_terms.append({
        'A': {(0,1): 1, (1,1): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(0,0): 1}
    })
    
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for t in range(7):
        term = strassen_terms[t]
        # For each of the 27 schoolbook terms in the 3x3x3 multiplication
        for si in range(3):  # sub-index for A row
            for sj in range(3):  # sub-index for A col / B row
                for sk in range(3):  # sub-index for B col / C col
                    # This is the schoolbook term for the 3x3 multiplication
                    # A_sub[si, sj] * B_sub[sj, sk] -> C_sub[si, sk]
                    
                    # A-part: for each block (bi, bj) with coefficient alpha_t[bi][bj]
                    for (bi, bj), alpha in term['A'].items():
                        a_row = 3 * bi + si
                        a_col = 3 * bj + sj
                        a_idx = a_row * m + a_col  # a_row * 6 + a_col
                        U[a_idx][col] = alpha
                    
                    # B-part: for each block (bi, bj) with coefficient beta_t[bi][bj]
                    for (bi, bj), beta in term['B'].items():
                        b_row = 3 * bi + sj
                        b_col = 3 * bj + sk
                        b_idx = b_row * p + b_col  # b_row * 6 + b_col
                        V[b_idx][col] = beta
                    
                    # C-part: for each block (ci, cj) with coefficient gamma_t[ci][cj]
                    for (ci, cj), gamma in term['C'].items():
                        c_row = 3 * ci + sk  # k index
                        c_col = 3 * cj + si  # i index
                        c_idx = c_row * n + c_col  # c_row * 6 + c_col
                        W[c_idx][col] = gamma
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
