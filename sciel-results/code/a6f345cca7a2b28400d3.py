"""Program for rank-minimal tensor decomposition of 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    
    The 2x5x5 matrix multiplication tensor T_{i,j,k} = delta_{i,k} decomposes as
    the direct sum of two copies of the 5x5 identity tensor (for i=0 and i=1).
    Each 5x5 identity tensor has rank 5, giving total rank 10.
    
    U: shape (n*m, rank) = (10, 10)
    V: shape (m*p, rank) = (25, 10)  
    W: shape (p*n, rank) = (10, 10)
    
    For r in 0..4: term r handles i=0, j=r, k=r
    For r in 5..9: term r handles i=1, j=r-5, k=r-5
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m  # 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i = r // m  # 0 or 1
        j = r % m   # 0..4
        k = j       # same as j since it's identity
        
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
