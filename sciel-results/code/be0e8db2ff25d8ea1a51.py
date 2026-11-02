import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    """
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen patterns for 2x2 matrix multiplication
    strassen_A = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ]
    
    strassen_B = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[0, -1], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ]
    
    strassen_C = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, -1]],  # M2
        [[0, 1], [0, 1]],   # M3
        [[1, 0], [1, 0]],   # M4
        [[-1, 1], [0, 0]],  # M5
        [[0, 0], [0, 1]],   # M6
        [[1, 0], [0, 0]],   # M7
    ]
    
    # For each block-level term t (0..6):
    # L_t is a 2x2 matrix where L_t[bi][bj] is a 2x2 matrix of coefficients for A blocks
    # Specifically, L_t[bi][bj] = strassen_A[t][bi][bj] * I (identity 2x2) if we're just selecting blocks
    # But actually, L_t[bi][bj] is a 2x2 matrix that represents the linear combination
    
    # For block-level Strassen term t:
    # The left factor L_t is a 2x2 matrix of 2x2 blocks
    # L_t[bi][bj] is a 2x2 matrix where (dr, dc) entry = strassen_A[t][bi][bj] * delta(dr,dc)
    # Wait, that's not right either.
    
    # Let me think about this differently.
    # A is a 4x4 matrix. Split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Each Aij is a 2x2 matrix.
    # 
    # Block-level Strassen term t:
    # L_t is a 2x2 matrix where L_t[bi][bj] is a 2x2 matrix
    # L_t[bi][bj][dr][dc] = coefficient of A[2*bi+dr][2*bj+dc] in the left factor
    # 
    # From strassen_A[t][bi][bj], this is the coefficient for the entire block A_block(bi,bj)
    # So L_t[bi][bj][dr][dc] = strassen_A[t][bi][bj] * delta(dr,dc)? No.
    # 
    # Actually, strassen_A[t][bi][bj] is the coefficient for the block, meaning:
    # L_t[bi][bj] = strassen_A[t][bi][bj] * I_2 (2x2 identity)
    # So L_t[bi][bj][dr][dc] = strassen_A[t][bi][bj] if dr==dc, else 0? No, that's wrong.
    # 
    # I think the correct interpretation is:
    # L_t[bi][bj] is a 2x2 matrix where each entry (dr, dc) has coefficient strassen_A[t][bi][bj]
    # So L_t[bi][bj][dr][dc] = strassen_A[t][bi][bj] for all dr, dc
    # 
    # But that doesn't make sense dimensionally. Let me look at the original code again.
    
    # From the original program:
    # for (bi, bj), block_coeff in block_A.items():
    #     for (sr, sc), scalar_coeff in scalar_A.items():
    #         a_row = 2 * bi + sr
    #         a_col = 2 * bj + sc
    #         U[a_row * 4 + a_col, r_idx] += block_coeff * scalar_coeff
    #
    # So block_A is a dict mapping (bi, bj) -> block_coeff
    # scalar_A is a dict mapping (sr, sc) -> scalar_coeff
    # And the total coefficient is block_coeff * scalar_coeff
    #
    # For the (t,s) term, block_A = strassen_A[t] (as a dict), scalar_A = strassen_A[s] (as a dict)
    # So U[i*4+k, 7t+s] = strassen_A[t][i//2][k//2] * strassen_A[s][i%2][k%2]
    #
    # This is exactly what I had before! So my construction should be correct.
    
    for t in range(7):
        for s in range(7):
            r_idx = t * 7 + s
            
            for i in range(4):
                for k in range(4):
                    bi, bk = i // 2, k // 2
                    si, sk = i % 2, k % 2
                    U[i * 4 + k, r_idx] = strassen_A[t][bi][bk] * strassen_A[s][si][sk]
            
            for k in range(4):
                for j in range(4):
                    bk, bj = k // 2, j // 2
                    sk, sj = k % 2, j % 2
                    V[k * 4 + j, r_idx] = strassen_B[t][bk][bj] * strassen_B[s][sk][sj]
            
            for j in range(4):
                for i in range(4):
                    bi, bj = i // 2, j // 2
                    si, sj = i % 2, j % 2
                    W[j * 4 + i, r_idx] = strassen_C[t][bi][bj] * strassen_C[s][si][sj]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
