"""Matrix multiplication 2x2 by 2x6 - Strassen-based bilinear circuit synthesis."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm for 2x2x2 on each pair of columns
    # This gives rank 3 * 7 = 21
    
    # Strassen for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block of 2 columns (b = 0, 1, 2), columns are 2b and 2b+1
    for b in range(p // 2):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # 7 Strassen multiplications for this block
        for s in range(7):
            col = b * 7 + s
            
            # U coefficients (for A matrix, 4 entries: a00, a01, a10, a11)
            # Index in U: i*m + j = i*2 + j
            if s == 0:  # M1 = (a00+a11)(b00+b11)
                U[0][col] = 1   # a00
                U[3][col] = 1   # a11
            elif s == 1:  # M2 = (a10+a11)b00
                U[2][col] = 1   # a10
                U[3][col] = 1   # a11
            elif s == 2:  # M3 = a00(b10-b11)
                U[0][col] = 1   # a00
            elif s == 3:  # M4 = a11(b01-b00)
                U[3][col] = 1   # a11
            elif s == 4:  # M5 = (a00+a01)b11
                U[0][col] = 1   # a00
                U[1][col] = 1   # a01
            elif s == 5:  # M6 = (a10-a00)(b00+b01)
                U[2][col] = 1   # a10
                U[0][col] = -1  # a00
            elif s == 6:  # M7 = (a01-a11)(b10+b11)
                U[1][col] = 1   # a01
                U[3][col] = -1  # a11
            
            # V coefficients (for B matrix, 12 entries: b00..b05, b10..b15)
            # Index in V: j*p + k = j*6 + k
            if s == 0:  # (b00+b11)
                V[0 * p + k0][col] = 1   # b0k0
                V[1 * p + k1][col] = 1   # b1k1
            elif s == 1:  # b00
                V[0 * p + k0][col] = 1   # b0k0
            elif s == 2:  # (b10-b11)
                V[1 * p + k0][col] = 1   # b1k0
                V[1 * p + k1][col] = -1  # b1k1
            elif s == 3:  # (b01-b00)
                V[0 * p + k1][col] = 1   # b0k1
                V[0 * p + k0][col] = -1  # b0k0
            elif s == 4:  # b11
                V[1 * p + k1][col] = 1   # b1k1
            elif s == 5:  # (b00+b01)
                V[0 * p + k0][col] = 1   # b0k0
                V[0 * p + k1][col] = 1   # b0k1
            elif s == 6:  # (b10+b11)
                V[1 * p + k0][col] = 1   # b1k0
                V[1 * p + k1][col] = 1   # b1k1
            
            # W coefficients (for C reconstruction, 12 entries: c00..c05, c10..c15)
            # Index in W: k*n + i = k*2 + i
            if s == 0:  # M1 contributes to C00 (+1), C11 (+1)
                W[k0 * n + 0][col] = 1   # C[0,k0]
                W[k1 * n + 1][col] = 1   # C[1,k1]
            elif s == 1:  # M2 contributes to C10 (+1), C11 (-1)
                W[k0 * n + 1][col] = 1   # C[1,k0]
                W[k1 * n + 1][col] = -1  # C[1,k1]
            elif s == 2:  # M3 contributes to C01 (+1), C11 (+1)
                W[k1 * n + 0][col] = 1   # C[0,k1]
                W[k1 * n + 1][col] = 1   # C[1,k1]
            elif s == 3:  # M4 contributes to C00 (+1), C10 (+1)
                W[k0 * n + 0][col] = 1   # C[0,k0]
                W[k0 * n + 1][col] = 1   # C[1,k0]
            elif s == 4:  # M5 contributes to C00 (-1), C01 (+1)
                W[k0 * n + 0][col] = -1  # C[0,k0]
                W[k1 * n + 0][col] = 1   # C[0,k1]
            elif s == 5:  # M6 contributes to C11 (+1)
                W[k1 * n + 1][col] = 1   # C[1,k1]
            elif s == 6:  # M7 contributes to C00 (+1)
                W[k0 * n + 0][col] = 1   # C[0,k0]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
