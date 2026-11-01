"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time

def construct():
    """Find rank-minimal exact tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Schoolbook decomposition as baseline: rank = n*m*p = 52
    rank_schoolbook = n * m * p
    
    # Try to find a better decomposition using known techniques
    # For 2x2 * 2x13, we can potentially use a decomposition that shares
    # some products across the 13 columns since A is shared.
    
    # Approach: Use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
    # We can try to find a rank-r decomposition where r < 52.
    
    # For now, return the schoolbook decomposition.
    # A more sophisticated search would try to find linear dependencies
    # that allow rank reduction.
    
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
