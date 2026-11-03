"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 66 (42 from Strassen + 24 from schoolbook)
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0+1) * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M2
        U[(i0+1) * m + j0][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        
        # M3
        U[i0 * m + j0][col] = 1
        V[(j0+1) * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M4
        U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0+1)][col] = 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k0+1) * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M6
        U[(i0+1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + i0][col] = -1
        W[k0 * n + (i0+1)][col] = 1
        col += 1
        
        # M7
        U[i0 * m + (j0+1)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1

    # Apply Strassen to 2x2x2 blocks for i in {0,1}
    # i0=0, j0 in {0,2}, k0 in {0,2,4}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
            
    # Apply schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
