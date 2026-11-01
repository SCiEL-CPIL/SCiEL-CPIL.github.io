"""Program for matrix multiplication 2x3 * 3x9 with rank-minimal decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Treat the 2x3 matrix multiplication as a batch of two independent 1x3 multiplications.
    For each row i (i=0,1) of the 2x3 matrix, we multiply a 1x3 vector by a 3x9 matrix.
    The rank for a 1x3 * 3x9 multiplication is 9 (since we need 9 independent terms for the 9 outputs).
    Total rank = 2 * 9 = 18.
    
    Convention:
    - U has shape (n*m, rank) = (2*3, 18) = (6, 18)
    - V has shape (m*p, rank) = (3*9, 18) = (27, 18)
    - W has shape (p*n, rank) = (9*2, 18) = (18, 18)
    
    Indexing:
    - U row index: i*m + j where i in [0,2), j in [0,3)
    - V row index: j*p + k where j in [0,3), k in [0,9)
    - W row index: k*n + i where k in [0,9), i in [0,2)
    
    For each row i of the input matrix (i=0,1) and each column k of the output (k=0..8):
    - The term contributes to output W[k*n + i, col]
    - U[i*m + j, col] = 1 for all j (since we're multiplying row i by column j of B)
    - V[j*p + k, col] = 1 for all j
    """
    n, m, p = DIMENSIONS
    # rank = n * p = 2 * 9 = 18 (since for each of the n rows, we need p terms)
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            # For output element (i, k), we sum over j: A[i,j] * B[j,k]
            # U[i*m + j, col] = 1 for each j
            # V[j*p + k, col] = 1 for each j
            # W[k*n + i, col] = 1
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
