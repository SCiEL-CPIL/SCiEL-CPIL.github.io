"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen parameters for 2x2
    # U coefficients (A_ij)
    # V coefficients (B_jk)
    # W coefficients (C_ki)
    
    # Block 0: k in {0, 1}
    # Block 1: k in {2, 3}
    # Block 2: k in {4, 5}
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00+A11)(B0k0+B1k1)
        c = t * 7 + 0
        U[0][c] = 1; U[3][c] = 1
        V[0][c] = 1; V[1 * p + k1][c] = 1
        W[k0 * n + 0][c] = 1; W[k1 * n + 1][c] = 1
        
        # M2: (A10+A11)B0k0
        c = t * 7 + 1
        U[2][c] = 1; U[3][c] = 1
        V[0][c] = 1
        W[k0 * n + 1][c] = 1; W[k1 * n + 1][c] = -1
        
        # M3: A00(B1k0-B1k1)
        c = t * 7 + 2
        U[0][c] = 1
        V[1 * p + k0][c] = 1; V[1 * p + k1][c] = -1
        W[k1 * n + 0][c] = 1; W[k1 * n + 1][c] = 1
        
        # M4: A11(B0k1-B0k0)
        c = t * 7 + 3
        U[3][c] = 1
        V[0 * p + k1][c] = 1; V[0 * p + k0][c] = -1
        W[k0 * n + 0][c] = -1; W[k1 * n + 0][c] = 1
        
        # M5: (A00+A01)B1k1
        c = t * 7 + 4
        U[0][c] = 1; U[1][c] = 1
        V[1 * p + k1][c] = 1
        W[k0 * n + 0][c] = 1
        
        # M6: (A10-A00)(B0k0+B0k1)
        c = t * 7 + 5
        U[2][c] = 1; U[0][c] = -1
        V[0 * p + k0][c] = 1; V[0 * p + k1][c] = 1
        W[k1 * n + 1][c] = 1
        
        # M7: (A01-A11)(B1k0+B1k1)
        c = t * 7 + 6
        U[1][c] = 1; U[3][c] = -1
        V[1 * p + k0][c] = 1; V[1 * p + k1][c] = 1
        W[k0 * n + 0][c] = -1; W[k0 * n + 1][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
