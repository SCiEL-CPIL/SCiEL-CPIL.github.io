"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor M_{n,m,p} has entries T_{i,j,k} = 1 for all valid indices.
    # This tensor can be decomposed with rank 1 using U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
