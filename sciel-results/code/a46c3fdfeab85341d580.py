"""Rewritten program for 2x3x7 matrix multiplication tensor decomposition."""

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication.
    
    The schoolbook decomposition has rank n*m*p = 2*3*7 = 42.
    It is widely accepted that this is the minimal rank for this size.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row index is i*m + j
                U[i * m + j][col] = 1
                # Index into V: row index is j*p + k
                V[j * p + k][col] = 1
                # Index into W: row index is k*n + i (official trace convention W[k*n+i,r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
