"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        U[0][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[4+k1][col] = 1
        W[k0][col] = 1
        W[k1+3][col] = 1
        col += 1
        
        U[3][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        W[k0+1][col] = 1
        W[k1+3][col] = -1
        col += 1
        
        U[0][col] = 1
        V[k0+1][col] = 1
        V[4+k1][col] = -1
        W[k0+1][col] = 1
        W[k1+3][col] = 1
        col += 1
        
        U[4][col] = 1
        V[4+k0][col] = 1
        V[k0][col] = -1
        W[k0][col] = 1
        W[k0+1][col] = 1
        col += 1
        
        U[0][col] = 1
        U[1][col] = 1
        V[4+k1][col] = 1
        W[k0][col] = -1
        W[k0+1][col] = 1
        col += 1
        
        U[3][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[k0+1][col] = 1
        W[k1+3][col] = 1
        col += 1
        
        U[1][col] = 1
        U[4][col] = -1
        V[4+k0][col] = 1
        V[4+k1][col] = 1
        W[k0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[8+k0][col] = 1
        W[k0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[8+k1][col] = 1
        W[k0+1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[8+k0][col] = 1
        W[k0+4][col] = 1
        col += 1
        
        U[5][col] = 1
        V[8+k1][col] = 1
        W[k0+5][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
