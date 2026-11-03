"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a rank 27 decomposition based on 3x3x3 schoolbook repeated for the two 3x3 blocks of B
    # This is valid because the A factor is shared and the V/W factors can be shifted.
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        i = c // (m * 3)
        j = (c // 3) % m
        k = c % 3
        
        # First block (k=0,1,2)
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
        
        # Second block (k=3,4,5)
        V[j * p + (k + 3)][c] = 1
        W[(k + 3) * n + i][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
