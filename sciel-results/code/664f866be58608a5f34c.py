"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-10 tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    The 2x4 matrix A has rows A_0 (1x4) and A_1 (1x4).
    Each row is multiplied by the 4x5 matrix B.
    
    For a 1x4 by 4x5 multiplication, the tensor has rank 5 (schoolbook):
    For each output column k (0..4), we compute sum_j A_j * B[j,k].
    
    So for the full 2x4 by 4x5 multiplication:
    - Row 0 of A: 5 terms (one per output column)
    - Row 1 of A: 5 terms (one per output column)
    Total rank = 10.
    
    Indexing conventions:
    - A is 2x4: A[i, j] where i in 0..1, j in 0..3
    - B is 4x5: B[j, k] where j in 0..3, k in 0..4
    - C is 2x5: C[i, k] where i in 0..1, k in 0..4
    
    U has shape (n*m, rank) = (8, 10): indexed by [i*m+j, c]
    V has shape (m*p, rank) = (20, 10): indexed by [j*p+k, c]
    W has shape (p*n, rank) = (10, 10): indexed by [k*n+i, c]
    
    For each (i, k) pair (10 combinations), we create one rank-1 term:
    c = i * 5 + k
    
    U[i*4 + j, c] = 1 for all j (since A[i, j] is multiplied by B[j, k])
    V[j*5 + k, c] = 1 for all j
    W[k*2 + i, c] = 1
    """
    n, m, p = 2, 4, 5
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            for j in range(m):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
