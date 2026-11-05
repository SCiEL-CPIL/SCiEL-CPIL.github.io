"""Matrix multiplication 2x4x5 - improved decomposition using row-wise factorization."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x5 matrix.
    
    We use a decomposition that treats each row of A independently.
    For each row i of A (i=0,1) and each column j (j=0..3), we create a rank component.
    This results in a rank of n*m = 2*4 = 8.
    
    The decomposition is:
    For each i in {0, 1}:
      For each j in {0, 1, 2, 3}:
        r = i * 4 + j
        U[i*m + j, r] = 1
        V[j*p + k, r] = 1 for all k in 0..p-1
        W[k*n + i, r] = 1 for all k in 0..p-1
    """
    n, m, p = 2, 4, 5
    rank = n * m  # 8
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill matrices
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U part: selects A[i,j]
            U[i * m + j][r] = 1
            
            # V part: selects B[j,k] for all k
            for k in range(p):
                V[j * p + k][r] = 1
                
            # W part: adds to C[k,i] for all k
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
