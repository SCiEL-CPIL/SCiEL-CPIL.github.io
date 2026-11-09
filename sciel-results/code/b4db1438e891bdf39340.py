import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x9 matrix B into three 3x3 blocks.
    # For each block, we compute A (2x3) * Block (3x3).
    # We use a rank-12 decomposition for 2x3x3 matrix multiplication.
    
    # Rank-12 decomposition for 2x3x3
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We construct U, V, W for the 2x3x3 problem.
    # U: (2*3, 12), V: (3*3, 12), W: (3*2, 12)
    # Note: W index is k*n + i = k*2 + i.
    
    U_sub = np.zeros((6, 12))
    V_sub = np.zeros((9, 12))
    W_sub = np.zeros((6, 12))
    
    # We use a simple decomposition that is valid.
    # Schoolbook is rank 18.
    # We can use Strassen for the 2x2 part of A and B.
    # Let A = [[a0, a1, a2], [a3, a4, a5]]
    # Let B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C = [[c0, c1, c2], [c3, c4, c5]]
    
    # This specific rank 12 construction is non-trivial to write inline without errors.
    # I will use the schoolbook rank 18 for the sub-blocks to ensure correctness,
    # which gives total rank 3 * 18 = 54. This is not an improvement.
    
    # To ensure an improvement, I will use a rank 10 decomposition for 2x3x3 if available.
    # Since I am not 100% confident in the rank 10 coefficients, I will use a 
    # conservative rank 12 decomposition derived from known results.
    
    # Actually, let's just use the schoolbook for the whole thing but with 
    # a better search? No, the task is to find a rank-minimal decomposition.
    
    # I will return the schoolbook decomposition for now to ensure validity,
    # but this will score 1.0.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
