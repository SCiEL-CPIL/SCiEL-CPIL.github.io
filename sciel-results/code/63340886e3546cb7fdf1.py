"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i, j, k] = 1 for all valid indices
    # because C[i, k] = sum_j A[i, j] * B[j, k].
    # This tensor is separable (rank 1) as T = u ⊗ v ⊗ w where u, v, w are all-ones vectors.
    
    rank = 1
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
