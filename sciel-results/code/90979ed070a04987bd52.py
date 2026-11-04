"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication C = AB is defined by T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k].
    # This tensor can be decomposed as rank 1:
    # U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1
    # Sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 * 1 * 1 = 1 = T[i,j,k].
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
