"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using Strassen's algorithm for 2x2 blocks
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p + k
    # W indices: k*n + i
    
    for t in range(3):
        # t corresponds to block t of B (columns 2t, 2t+1)
        # e_t = B[0, 2t], f_t = B[0, 2t+1]
        # g_t = B[1, 2t], h_t = B[1, 2t+1]
        
        # M1 = (a+d)(e+h)
        col = t * 7
        U[0][col] = 1
        U[3][col] = 1
        V[0 * p + 2 * t][col] = 1
        V[1 * p + 2 * t + 1][col] = 1
        W[2 * t * n + 0][col] = 1
        W[(2 * t + 1) * n + 1][col] = 1
        
        # M2 = (c+d)e
        col = t * 7 + 1
        U[2][col] = 1
        U[3][col] = 1
        V[0 * p + 2 * t][col] = 1
        W[2 * t * n + 1][col] = 1
        W[(2 * t + 1) * n + 1][col] = -1
        
        # M3 = a(g-h)
        col = t * 7 + 2
        U[0][col] = 1
        V[1 * p + 2 * t][col] = 1
        V[1 * p + 2 * t + 1][col] = -1
        W[(2 * t + 1) * n + 0][col] = 1
        W[(2 * t + 1) * n + 1][col] = 1
        
        # M4 = d(f-e)
        col = t * 7 + 3
        U[3][col] = 1
        V[0 * p + 2 * t + 1][col] = 1
        V[0 * p + 2 * t][col] = -1
        W[2 * t * n + 0][col] = 1
        W[2 * t * n + 1][col] = 1
        
        # M5 = (a+b)h
        col = t * 7 + 4
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + 2 * t + 1][col] = 1
        W[2 * t * n + 0][col] = -1
        W[(2 * t + 1) * n + 0][col] = 1
        
        # M6 = (c-a)(e+f)
        col = t * 7 + 5
        U[2][col] = 1
        U[0][col] = -1
        V[0 * p + 2 * t][col] = 1
        V[0 * p + 2 * t + 1][col] = 1
        W[(2 * t + 1) * n + 1][col] = 1
        
        # M7 = (b-d)(g+h)
        col = t * 7 + 6
        U[1][col] = 1
        U[3][col] = -1
        V[1 * p + 2 * t][col] = 1
        V[1 * p + 2 * t + 1][col] = 1
        W[2 * t * n + 0][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
