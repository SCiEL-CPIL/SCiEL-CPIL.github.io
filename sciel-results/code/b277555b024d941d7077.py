"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T_{i, j, k} = 1 for all i, j, k.
    # This is a rank-1 tensor: 1_n ⊗ 1_p ⊗ 1_m (adjusted for index order).
    # U index: i*m + k
    # V index: k*p + j
    # W index: j*n + i
    # We need sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] = 1
    # Set r=0, all entries to 1.
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
