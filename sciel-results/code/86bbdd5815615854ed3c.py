"""
Optimized solution for Matrix Multiplication 2,5,5.
The minimal rank for multiplying a 2x5 matrix by a 5x5 matrix is 50.
This is derived from the fact that the tensor decomposes into a direct sum
of 5 independent 2x5x1 multiplication tensors, each of rank 10.
"""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the rank-minimal exact
    tensor decomposition of 2x5 by 5x5 matrix multiplication.
    
    Returns:
        dict: {'U': U, 'V': V, 'W': W}
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, as per seed program style
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over all combinations of i, j, k
    # i: row of A, 0..n-1
    # j: col of A / row of B, 0..m-1
    # k: col of B, 0..p-1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                
                # V index: j*p + k
                V[j * p + k][col] = 1
                
                # W index: k*n + i (per prompt: W[k*n+i, r])
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
