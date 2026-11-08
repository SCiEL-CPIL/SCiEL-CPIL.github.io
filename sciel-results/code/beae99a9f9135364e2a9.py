"""Program for matrix multiplication 2x4 by 4x6 with rank-minimal decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-8 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    The decomposition treats each row of the first matrix independently.
    Row i of C = Row i of A * B
    This requires rank m = 4 per row, total rank 8.
    
    Indexing:
    - U has shape (n*m, rank) = (8, 8), indexed by (i, j) where i in [0,1], j in [0,3]
    - V has shape (m*p, rank) = (24, 8), indexed by (j, k) where j in [0,3], k in [0,5]
    - W has shape (p*n, rank) = (12, 8), indexed by (k, i) where k in [0,5], i in [0,1]
    
    Trace convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 2 * m  # rank = 8
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 8)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 8)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 8)
    
    # For each row i of A, and each column j of A (which corresponds to column k of B):
    # C[i][k] += A[i][j] * B[j][k]
    # 
    # Decomposition: for each (i, j), we have one rank-1 term
    # U[i*m + j, r] = 1 where r = i*m + j (the term index for this (i,j) pair)
    # V[j*p + k, r] = 1 for all k (but actually V depends on j and k)
    # 
    # Wait, let me reconsider. For a fixed i:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # 
    # For term r = i*m + j (i.e., r encodes which row i and which column j of A):
    # U[i*m + j, r] = 1 (selects A[i][j])
    # V[j*p + k, r] = 1 (selects B[j][k] for this fixed j)
    # W[k*n + i, r] = 1 (places result at C[i][k])
    #
    # So for r = i*m + j:
    # - U[row i, col j] (i.e., U[i*m + j, r]) = 1
    # - V[row j, col k] (i.e., V[j*p + k, r]) = 1 for all k in [0, p-1]
    # - W[row k, col i] (i.e., W[k*n + i, r]) = 1 for all k in [0, p-1]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j  # term index, r in [0, 7]
            # U: index (i, j) -> row i*m + j
            U[i * m + j][r] = 1
            # V: for this term, we need V[j*p + k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
