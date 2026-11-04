"""Program for 2x3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Direct tensor decomposition for 2x3x6 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0..5}
    
    We use a direct construction where each rank-1 tensor computes one product A[i,j]*B[j,k]
    and places it at the correct position in C. This gives rank = n*m*p = 36.
    
    However, we can do better by observing that for fixed (i,j), we need to compute
    A[i,j]*B[j,k] for all k. We can share the A[i,j] factor.
    
    Let's try a decomposition with rank 24:
    For each i in {0,1} and j in {0,1,2}, we compute 4 products that cover all 6 k values
    using linear combinations.
    """
    n, m, p = 2, 3, 6
    
    # Start with trivial rank 36 and see if we can reduce
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
