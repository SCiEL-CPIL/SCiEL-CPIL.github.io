"""Matrix multiplication 3,3,4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication.
    
    The multiplication tensor T has entries T[i,j,k] = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    where the output index follows W[k*n+i, r].
    
    We use the schoolbook decomposition with rank = n*m*p = 36.
    Each term corresponds to one (i, j, k) triple.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36
    
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
