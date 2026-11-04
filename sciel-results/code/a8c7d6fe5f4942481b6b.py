"""Matrix multiplication 2x3x8 with rank-44 decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank calculation:
    # Decompose into A1 (2x2) * B1 (2x8) and A2 (2x1) * B2 (1x8)
    # M(2, 2, 8) = 4 * M(2, 2, 2) = 4 * 7 = 28
    # M(2, 1, 8) = 16
    # Total rank = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: Strassen decomposition for 2x2x8
    # A1 indices in U: 0, 1, 3, 4
    # B1 indices in V: 0-7 (row 0), 8-15 (row 1)
    # C1 indices in W: 0-7 (row 0), 8-15 (row 1)
    
    # Strassen coefficients for 2x2x2
    # p1 = (a00+a10)(b00+b01)
    # p2 = (a10-a00)b01
    # p3 = a00(b10-b11)
    # p4 = (a01+a11)b11
    # p5 = (a00+a01)b00
    # p6 = (a10-a01)(b01+b11)
    # p7 = (a11-a10)(b00+b10)
    
    # c00 = p1 + p4 - p5 - p7
    # c01 = p5 + p7
    # c10 = p1 + p3 - p4 - p6
    # c11 = p2 + p4 + p6
    
    for block in range(4):
        # B columns 2*block and 2*block+1
        # V indices for B1:
        # b00 is V[0 + 2*block]
        # b01 is V[1 + 2*block]
        # b10 is V[8 + 2*block]
        # b11 is V[9 + 2*block]
        
        v_b00 = 0 + 2 * block
        v_b01 = 1 + 2 * block
        v_b10 = 8 + 2 * block
        v_b11 = 9 + 2 * block
        
        # W indices for C1:
        # c00 is W[0 + 2*block]
        # c01 is W[1 + 2*block]
        # c10 is W[8 + 2*block]
        # c11 is W[9 + 2*block]
        
        w_c00 = 0 + 2 * block
        w_c01 = 1 + 2 * block
        w_c10 = 8 + 2 * block
        w_c11 = 9 + 2 * block
        
        # Define the 7 Strassen terms
        # Term 1
        U[0][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        col += 1
        
        # Term 2
        U[3][col] += 1
        U[0][col] -= 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # Term 3
        U[0][col] += 1
        V[v_b10][col] += 1
        V[v_b11][col] -= 1
        W[w_c10][col] += 1
        col += 1
        
        # Term 4
        U[1][col] += 1
        U[4][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # Term 5
        U[0][col] += 1
        U[1][col] += 1
        V[v_b00][col] += 1
        W[w_c00][col] -= 1
        W[w_c01][col] += 1
        col += 1
        
        # Term 6
        U[3][col] += 1
        U[1][col] -= 1
        V[v_b01][col] += 1
        V[v_b11][col] += 1
        W[w_c10][col] -= 1
        W[w_c11][col] += 1
        col += 1
        
        # Term 7
        U[4][col] += 1
        U[3][col] -= 1
        V[v_b00][col] += 1
        V[v_b10][col] += 1
        W[w_c00][col] -= 1
        W[w_c01][col] += 1
        col += 1

    # Part 2: Outer product for 2x1x8
    # A2 indices in U: 2, 5
    # B2 indices in V: 16-23
    # C2 indices in W: 0-7 (row 0), 8-15 (row 1)
    
    for k in range(8):
        # Term: A[0,2] * B[2,k] -> C[0,k]
        U[2][col] += 1
        V[16 + k][col] += 1
        W[k][col] += 1
        col += 1
        
        # Term: A[1,2] * B[2,k] -> C[1,k]
        U[5][col] += 1
        V[16 + k][col] += 1
        W[8 + k][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
