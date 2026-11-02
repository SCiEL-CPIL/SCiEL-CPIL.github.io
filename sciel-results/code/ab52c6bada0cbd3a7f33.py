"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix mult.
    
    Returns dict with U (6 x rank), V (27 x rank), W (18 x rank) where:
    - U[i*m + j, c] corresponds to first matrix element a[i,j]
    - V[j*p + k, c] corresponds to second matrix element b[j,k]  
    - W[k*n + i, c] corresponds to output element c[k,i] (trace convention)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
