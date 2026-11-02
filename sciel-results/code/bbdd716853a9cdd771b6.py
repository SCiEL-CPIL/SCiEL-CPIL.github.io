"""Matrix multiplication 2x2x12: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-48 exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x12 matrix.
    
    The decomposition uses the schoolbook structure: each of the 12 columns of
    the B matrix is multiplied independently by the 2x2 matrix A. Each such
    multiplication is the rank-4 tensor M_{2,2,1}, giving total rank 48.
    
    Index conventions:
      U: (n*m, rank) = (4, 48)  -- U[i*m + j, c]
      V: (m*p, rank) = (24, 48) -- V[j*p + k, c]  
      W: (p*n, rank) = (24, 48) -- W[k*n + i, c]
    
    For each (j, k) pair, we need 4 bilinear terms corresponding to the 4
    products of A's entries with B[j,k]. The column index c = j*12 + k*4 + t
    where t in {0,1,2,3} indexes the four A entries: A[0,0], A[0,1], A[1,0], A[1,1].
    
    Specifically:
      c = 0: A[0][0] * B[0][k] -> U[0][c]=1, V[0][c]=1, W[k*2+0][c]=1
      c = 1: A[0][1] * B[1][k] -> U[1][c]=1, V[12+k][c]=1, W[k*2+0][c]=1
      c = 2: A[1][0] * B[0][k] -> U[2][c]=1, V[0][c]=1, W[k*2+1][c]=1
      c = 3: A[1][1] * B[1][k] -> U[3][c]=1, V[12+k][c]=1, W[k*2+1][c]=1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each column k of B and each row j of B:
    #   We compute 4 products: A[i][j] * B[j][k] for i in {0,1}, j in {0,1}
    #   These contribute to output C[i][k]
    for k in range(p):  # column of B (and C)
        for j in range(m):  # row of B
            for i in range(n):  # row of C
                # Product A[i][j] * B[j][k] contributes to C[i][k]
                # Column index in the decomposition
                c = j * p + k * n + i  # (j,k,i) triple -> 0..47
                
                U[i * m + j][c] = 1  # A[i][j]
                V[j * p + k][c] = 1  # B[j][k]
                W[k * n + i][c] = 1  # C[i][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
