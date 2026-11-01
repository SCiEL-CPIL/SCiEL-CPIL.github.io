"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    For dimensions n x m x p, the rank is n*m*p.
    The factor matrices U, V, W have shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The entries are 0 or 1.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row i*m + j
                u_row = i * m + j
                # Index into V: row j*p + k
                v_row = j * p + k
                # Index into W: row k*n + i
                w_row = k * n + i
                
                U[u_row][column] = 1
                V[v_row][column] = 1
                W[w_row][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
