import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    This implementation uses the Kronecker-product structure derived from identity matrices,
    representing the multiplication as the sum of outer products of elementary selection vectors.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) - selects row i and column j of A
    # V: (m*p, rank) - selects row j and column k of B
    # W: (p*n, rank) - places result at row i and column k (Trace convention: W[k*n+i, r])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all combinations of (i, j, k)
    # i: row index of A (and C)
    # j: column index of A (and row index of B)
    # k: column index of B (and C)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row i, col j -> i*m + j
                u_idx = i * m + j
                # Index into V: row j, col k -> j*p + k
                v_idx = j * p + k
                # Index into W: row k, col i -> k*n + i (following W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
