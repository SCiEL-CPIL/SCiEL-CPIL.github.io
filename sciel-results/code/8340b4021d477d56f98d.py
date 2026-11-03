"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_right * B_bottom
    # A_right is col 2 of A (2x1). Indices in U: 2, 5.
    # B_bottom is row 2 of B (1x8). Indices in V: 16..23.
    # Output is 2x8.
    for i in range(n):
        for k in range(p):
            # U index for A[i, 2] is i*m + 2
            U[i * m + 2][col] = 1
            # V index for B[2, k] is 2*p + k
            V[2 * p + k][col] = 1
            # W index for C[i, k] is k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_left * B_top
    # A_left is cols 0,1 of A (2x2).
    # B_top is rows 0,1 of B (2x8).
    # Split B_top into 4 blocks of 2x2.
    for j in range(4):
        # Strassen for 2x2x2
        # X = A_left
        # x00 -> U[0], x01 -> U[1], x10 -> U[3], x11 -> U[4]
        # Y = B_top[:, 2j:2j+2]
        # y00 -> V[2j], y01 -> V[2j+1]
        # y10 -> V[8+2j], y11 -> V[9+2j]
        
        # M1 = (x00 + x11)(y00 + y11)
        U[0][col] = 1
        U[4][col] = 1
        V[2*j][col] = 1
        V[8+2*j][col] = 1
        # z00 += M1, z11 += M1
        W[4*j][col] = 1
        W[4*j+3][col] = 1
        col += 1
        
        # M2 = (x10 + x11) y00
        U[3][col] = 1
        U[4][col] = 1
        V[2*j][col] = 1
        # z10 += M2, z11 -= M2
        W[4*j+1][col] = 1
        W[4*j+3][col] = -1
        col += 1
        
        # M3 = x00 (y01 - y11)
        U[0][col] = 1
        V[2*j+1][col] = 1
        V[8+2*j+1][col] = -1
        # z01 += M3, z11 += M3
        W[4*j+2][col] = 1
        W[4*j+3][col] = 1
        col += 1
        
        # M4 = x11 (y10 - y00)
        U[4][col] = 1
        V[8+2*j][col] = 1
        V[2*j][col] = -1
        # z00 += M4, z10 += M4
        W[4*j][col] = 1
        W[4*j+1][col] = 1
        col += 1
        
        # M5 = (x00 + x01) y11
        U[0][col] = 1
        U[1][col] = 1
        V[8+2*j+1][col] = 1
        # z00 -= M5, z01 += M5
        W[4*j][col] = -1
        W[4*j+2][col] = 1
        col += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        U[3][col] = 1
        U[0][col] = -1
        V[2*j][col] = 1
        V[2*j+1][col] = 1
        # z11 += M6
        W[4*j+3][col] = 1
        col += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        U[1][col] = 1
        U[4][col] = -1
        V[8+2*j][col] = 1
        V[8+2*j+1][col] = 1
        # z00 += M7
        W[4*j][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
