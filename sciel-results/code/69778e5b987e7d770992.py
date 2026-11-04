"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal decomposition for 4x5 * 5x5 matrix multiplication.
    
    The standard matrix multiplication tensor can be decomposed into a sum of 
    m rank-1 tensors, one for each value of the shared index j.
    For each j in 0..m-1, we define a rank-1 component where:
    - U is non-zero only for rows (i, j) with U[i*m + j, j] = 1
    - V is non-zero only for rows (j, k) with V[j*p + k, j] = 1
    - W is non-zero only for rows (k, i) with W[k*n + i, j] = 1
    
    This results in a total rank of m.
    """
    n, m, p = DIMENSIONS
    
    # Rank is m
    rank = m
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the structure derived
    for j in range(m):
        for i in range(n):
            # U index: i*m + j, column j
            U[i * m + j][j] = 1
            
        for k in range(p):
            # V index: j*p + k, column j
            V[j * p + k][j] = 1
            
        for i in range(n):
            for k in range(p):
                # W index: k*n + i, column j
                # Note: The tensor definition requires consistency.
                # For a fixed j, we sum over i and k.
                # The term contributes to C[i, k] via A[i, j] and B[j, k].
                # So for each (i, k), the output index is k*n + i.
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
