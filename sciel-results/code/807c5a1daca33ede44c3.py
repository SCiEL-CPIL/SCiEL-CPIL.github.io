"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    # Indices:
    # A: 0:a, 1:b, 2:c, 3:d
    # B: 0:e, 1:f, 2:g, 3:h
    # C: 0:c00, 1:c10, 2:c01, 3:c11 (using W[k*n+i] with n=2)
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1 = (a+d)(e+h) -> c00 + c11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e -> c10 - c11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h) -> c01 + c11
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-f) -> c00 + c10
    U[3][3] = 1
    V[2][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h -> -c00 + c01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)(e+f) -> c11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h) -> c00
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
