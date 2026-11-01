"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(3):
        base_col = 7 * t
        # M1
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[2*t][base_col] = 1
        V[7+2*t][base_col] = 1
        W[4*t][base_col] = 1
        W[4*t+3][base_col] = 1
        
        # M2
        U[2][base_col+1] = 1
        U[3][base_col+1] = 1
        V[2*t][base_col+1] = 1
        W[4*t+1][base_col+1] = 1
        W[4*t+3][base_col+1] = -1
        
        # M3
        U[0][base_col+2] = 1
        V[2*t+1][base_col+2] = 1
        V[7+2*t][base_col+2] = -1
        W[4*t+2][base_col+2] = 1
        W[4*t+3][base_col+2] = 1
        
        # M4
        U[3][base_col+3] = 1
        V[6+2*t][base_col+3] = 1
        V[2*t][base_col+3] = -1
        W[4*t][base_col+3] = 1
        W[4*t+1][base_col+3] = 1
        
        # M5
        U[0][base_col+4] = 1
        U[1][base_col+4] = 1
        V[7+2*t][base_col+4] = 1
        W[4*t][base_col+4] = -1
        W[4*t+2][base_col+4] = 1
        
        # M6
        U[2][base_col+5] = 1
        U[0][base_col+5] = -1
        V[2*t][base_col+5] = 1
        V[2*t+1][base_col+5] = 1
        W[4*t+3][base_col+5] = 1
        
        # M7
        U[1][base_col+6] = 1
        U[3][base_col+6] = -1
        V[6+2*t][base_col+6] = 1
        V[7+2*t][base_col+6] = 1
        W[4*t][base_col+6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
