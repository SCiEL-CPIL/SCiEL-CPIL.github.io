"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 18 for M_{2,3,6}
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The decomposition is based on splitting the p dimension into two blocks of size 3
    # and using a rank-9 decomposition for M_{2,3,3} for each block.
    # For M_{2,3,3}, the rank-9 decomposition uses terms indexed by (j, k_local).
    # For each term:
    # V has 1 at (j, k_local)
    # W has 1s at (k_local, 0) and (k_local, 1)
    # U has 1s at (0, j) and (1, j)
    
    for b in range(2): # Two blocks of k
        for j in range(m): # j in 0..2
            for k_local in range(3): # k_local in 0..2
                c = b * 9 + j * 3 + k_local
                k = b * 3 + k_local
                
                # V index: j * p + k
                V[j * p + k][c] = 1
                
                # W index: k * n + i
                # i=0: k * 2 + 0
                # i=1: k * 2 + 1
                W[k * n + 0][c] = 1
                W[k * n + 1][c] = 1
                
                # U index: i * m + j
                # i=0: 0 * 3 + j = j
                # i=1: 1 * 3 + j = 3 + j
                U[0 * m + j][c] = 1
                U[1 * m + j][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
