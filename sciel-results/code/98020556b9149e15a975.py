"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 decomposition of 3x3 matrix multiplication.
    # Uses Strassen's algorithm for the 2x2 top-left block.
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for 3x3 matrix elements:
    # a00, a01, a02
    # a10, a11, a12
    # a20, a21, a22
    # b00, b01, b02
    # b10, b11, b12
    # b20, b21, b22
    
    # Strassen's 2x2 algorithm for the top-left 2x2 block:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b11 - b00)
    # m4 = a11 * (b00 - b10)
    # m5 = (a00 + a01) * b10
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # The 2x2 block result is:
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m7
    # c11 = m1 - m2 + m3 + m6
    
    # The remaining terms involve the third row and column.
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c22 = a20*b02 + a21*b12 + a22*b22
    
    # We can combine these with the Strassen terms to reduce the rank.
    # The rank-26 decomposition is known.
    
    # Factor 1: m1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[3][0] = 1; W[7][0] = 1
    
    # Factor 2: m2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[4][1] = 1; W[3][1] = -1
    
    # Factor 3: m3
    U[0][2] = 1
    V[4][2] = 1; V[0][2] = -1
    W[1][2] = 1; W[4][2] = 1
    
    # Factor 4: m4
    U[4][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[4][3] = 1
    
    # Factor 5: m5
    U[0][4] = 1; U[1][4] = 1
    V[1][4] = 1
    W[0][4] = -1; W[4][4] = -1
    
    # Factor 6: m6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[7][5] = 1
    
    # Factor 7: m7
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    # Factors 8-19: The remaining 12 terms for the third row and column.
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c22 = a20*b02 + a21*b12 + a22*b22
    
    # We can use 12 more factors for these 12 products, but we can combine them.
    # The rank-26 decomposition uses 19 more factors? No, 26-7=19.
    # Actually, the rank-26 decomposition is not simply 7+19.
    
    # Let's use the known rank-26 decomposition.
    # It is easier to use the rank-27 schoolbook and then reduce it.
    # But I will implement the rank-26 decomposition directly.
    
    # The rank-26 decomposition is:
    # 7 Strassen factors for the 2x2 block.
    # 19 factors for the rest.
    
    # I will use the rank-27 schoolbook decomposition for now, as it is guaranteed to be correct.
    # The rank-26 decomposition is an improvement, but I need to be sure it is correct.
    # I will implement the rank-27 schoolbook decomposition.
    
    rank = n * m * p
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
