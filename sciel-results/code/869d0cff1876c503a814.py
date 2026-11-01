"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact decomposition of the (2,4,4) matrix multiplication tensor.
    
    Key insight: The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since the
    coefficient of A[i,j]*B[j,k] in C[i,k] = sum_j A[i,j]*B[j,k] is always 1).
    
    This is a rank-1 tensor! We can write T[i,j,k] = 1*1*1 for all i,j,k.
    
    Therefore, the minimal rank is 1, achieved by:
    U[i*m+j, 0] = 1 for all i,j
    V[j*p+k, 0] = 1 for all j,k
    W[k*n+i, 0] = 1 for all k,i
    """
    n, m, p = DIMENSIONS
    rank = 1
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
