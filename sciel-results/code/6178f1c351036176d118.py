"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 matrix times 5x5 matrix.
    We use the schoolbook decomposition, which gives rank 75.
    The tensor is T[i,j,k] = sum_{j',k'} A[i,j'] * B[j',k'] * C[k',i]
    Actually the matrix multiplication tensor for (n x m) * (m x p) has entries indexed by (i, j, k)
    where i in [0,n), j in [0,m), k in [0,p), and the "multiplication" is sum_{l} A[i,l] * B[l,k].
    
    The standard decomposition has one term for each (i, l, k) triple, where l is the summation index.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    For term (i, l, k):
    U[i*m + l, term_idx] = 1
    V[l*p + k, term_idx] = 1
    W[k*n + i, term_idx] = 1  (following the trace convention W[k*n+i, r])
    
    Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    So W is indexed by [k*n+i, r].
    
    rank = n*m*p = 3*5*5 = 75
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U[i * m + l][col] = 1
                V[l * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
