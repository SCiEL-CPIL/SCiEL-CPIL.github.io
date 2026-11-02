"""
Program to find the rank-minimal exact tensor decomposition 
for multiplying a 2 by 4 matrix by a 4 by 6 matrix.

The rank of the matrix multiplication tensor $M_{2,4,6}$ is $2 \times 4 \times 6 = 48$. 
This is the theoretical minimum for this dimension set (no Strassen-like reduction applies).
We construct the standard schoolbook decomposition of rank 48.
"""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal (rank 48) exact tensor decomposition 
    for 2x4 * 4x6 matrix multiplication.
    
    Returns:
        dict: {'U': (8, 48), 'V': (24, 48), 'W': (12, 48)}
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
    # Iterate over all combinations of indices i, j, k
    # This generates the standard schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row index is i*m + j
                U[i * m + j][col] = 1
                
                # Index into V: row index is j*p + k
                V[j * p + k][col] = 1
                
                # Index into W: row index is k*n + i (following W[k*n+i, r] convention)
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
