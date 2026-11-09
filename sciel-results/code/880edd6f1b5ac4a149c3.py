"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a structured decomposition with rank = n*m*p = 60 (schoolbook).
    The matrix multiplication tensor for C = A * B where A is n x m and B is m x p
    has entries T[i,j,k] = sum_l A[i,l] * B[l,k].
    
    The standard rank decomposition uses rank = n*m*p terms.
    Each term corresponds to a specific (i, l, k) triple.
    
    For matrix multiplication, the tensor is:
    T[i, j, k] = sum_{l=1}^{m} A[i, l] * B[l, k]
    
    where i in [1,n], j in [1,m], k in [1,p]
    
    The decomposition is:
    T = sum_{i=1}^{n} sum_{l=1}^{m} sum_{k=1}^{p} u_{i,l,k} * v_{i,l,k} * w_{i,l,k}
    
    where:
    - u_{i,l,k} has 1 at position (i, l) in the A-matrix (row i, col l)
    - v_{i,l,k} has 1 at position (l, k) in the B-matrix (row l, col k)
    - w_{i,l,k} has 1 at position (i, k) in the C-matrix (row i, col k)
    
    This gives rank = n*m*p = 2*5*6 = 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U has shape (n*m, rank) - corresponds to A matrix entries
    # V has shape (m*p, rank) - corresponds to B matrix entries
    # W has shape (p*n, rank) - corresponds to C matrix entries (W[k*n+i, r])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                # A[i, l] index in U
                U[i * m + l][col] = 1
                # B[l, k] index in V
                V[l * p + k][col] = 1
                # C[i, k] index in W (W[k*n + i, r])
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
