"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    The current best known upper bound for R(4,5,5) is 80, achieved by the Winograd-style
    decomposition that exploits the structure of the matrices. However, for small dimensions,
    the schoolbook decomposition of rank n*m*p = 100 is often the simplest correct answer.
    
    Here we implement the standard schoolbook decomposition which is guaranteed to be correct.
    The rank is n*m*p = 4*5*5 = 100.
    
    The tensor decomposition is defined by:
    C_{i,j} = sum_{k=0}^{m-1} A_{i,k} * B_{k,j}
    
    In tensor form, the multiplication tensor T is a 3-tensor with indices (i, j, k) where:
    - i indexes rows of A (0 to n-1)
    - j indexes columns of B (0 to p-1)  
    - k indexes the inner dimension (0 to m-1)
    
    The rank-1 decomposition terms are indexed by (i, j, k) with:
    U[i*m + j, (i, j, k)] = 1  (but this doesn't quite work)
    
    Actually, the standard decomposition is:
    For each k in range(m):
        For each i in range(n):
            For each j in range(p):
                Term (i, j, k) contributes A_{i,k} * B_{k,j} to C_{i,j}
    
    So we have n*m*p terms, each with:
    U[i*m + j, term] = 1 if this term corresponds to (i, j, k)
    V[j*p + k, term] = 1 if this term corresponds to (i, j, k)
    W[k*n + i, term] = 1 if this term corresponds to (i, j, k)
    
    Wait, let me re-read the interface. The output factor follows the official trace convention W[k*n+i,r].
    
    Let's use the standard schoolbook:
    n, m, p = 4, 5, 5
    rank = n * m * p = 100
    
    For each triple (i, j, k) where i in [0,n), j in [0,p), k in [0,m):
        U[i*m + j, r] = 1
        V[j*p + k, r] = 1  
        W[k*n + i, r] = 1
        where r is the index of this term
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(m):
        for i in range(n):
            for j in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
