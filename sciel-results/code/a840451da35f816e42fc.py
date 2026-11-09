"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    # T_{2,3,3} = T_{2,2,2} (Strassen) + T_{2,2,1} + T_{2,1,3}
    # Indices:
    # A (2x3): 0,1,2,3,4,5
    # B (3x3): 0,1,2,3,4,5,6,7,8
    # C (2x3): 0,1,2,3,4,5
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. T_{2,2,2} part (Strassen)
    # A indices: 0,1,2,3 (A[0,0], A[0,1], A[1,0], A[1,1])
    # B indices: 0,1,3,4 (B[0,0], B[0,1], B[1,0], B[1,1])
    # C indices: 0,1,3,4 (C[0,0], C[0,1], C[1,0], C[1,1])
    
    # M1 = (a0+a3)(b0+b3)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (a2+a3)b0
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M3 = a0(b1-b3)
    U[0][col] = 1
    V[1][col] = 1; V[3][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = a3(b2-b0)
    U[3][col] = 1
    V[2][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (a0+a1)b3
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (a2-a0)(b0+b1)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (a1-a3)(b2+b3)
    U[1][col] = 1; U[3][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[4][col] = -1
    col += 1
    
    # 2. T_{2,2,1} part
    # A indices: 0,1,2,3
    # B indices: 5,7 (B[0,2], B[1,2])
    # C indices: 2,4 (C[0,2], C[1,2])
    
    # c0 += a0*b5 + a1*b7
    U[0][col] = 1
    V[5][col] = 1
    W[2][col] = 1
    col += 1
    
    U[1][col] = 1
    V[7][col] = 1
    W[2][col] = 1
    col += 1
    
    # c1 += a2*b5 + a3*b7
    U[2][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    U[3][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # 3. T_{2,1,3} part
    # A indices: 4,5 (A[0,2], A[1,2])
    # B indices: 6,7,8 (B[2,0], B[2,1], B[2,2])
    # C indices: 0,1,2,3,4,5
    
    # c00 += a4*b6
    U[4][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # c01 += a4*b7
    U[4][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    
    # c02 += a4*b8
    U[4][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # c10 += a5*b6
    U[5][col] = 1
    V[6][col] = 1
    W[3][col] = 1
    col += 1
    
    # c11 += a5*b7
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # c12 += a5*b8
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
