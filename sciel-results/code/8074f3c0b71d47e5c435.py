"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    def strassen_2x2():
        U = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W = [
            [1, 0, 0, 0, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 0, 0, 0, 1, 0]
        ]
        return U, V, W

    # Rank 49 decomposition for 4x4x4
    def strassen_4x4():
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        u2, v2, w2 = strassen_2x2()
        
        # Strassen formulas for 4x4x4
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A21)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # We need to map the 2x2 block indices to 4x4 indices.
        # A blocks: A11 (0-3), A12 (4-7), A21 (8-11), A22 (12-15)
        # B blocks: B11 (0-3), B12 (4-7), B21 (8-11), B22 (12-15)
        
        # Term 1: A11+A22, B11+B22
        # U: A11 (0-3) + A22 (12-15)
        # V: B11 (0-3) + B22 (12-15)
        # W: C11 (0-3) + C22 (12-15)
        for r in range(4):
            for c in range(7):
                U[r][c] += u2[r][c]
                U[r+12][c] += u2[r][c]
                V[r][c] += v2[r][c]
                V[r+12][c] += v2[r][c]
                W[r][c] += w2[r][c]
                W[r+12][c] += w2[r][c]

        # Term 2: A21+A22, B11
        # U: A21 (8-11) + A22 (12-15)
        # V: B11 (0-3)
        # W: C21 (8-11)
        for r in range(4):
            for c in range(7):
                U[r+8][c] += u2[r][c]
                U[r+12][c] += u2[r][c]
                V[r][c] += v2[r][c]
                W[r+8][c] += w2[r][c]

        # Term 3: A11, B12-B22
        # U: A11 (0-3)
        # V: B12 (4-7) - B22 (12-15)
        # W: C12 (4-7)
        for r in range(4):
            for c in range(7):
                U[r][c+7] += u2[r][c]
                V[r+4][c+7] += v2[r][c]
                V[r+12][c+7] -= v2[r][c]
                W[r+4][c+7] += w2[r][c]

        # Term 4: A22, B21-B11
        # U: A22 (12-15)
        # V: B21 (8-11) - B11 (0-3)
        # W: C21 (8-11)
        for r in range(4):
            for c in range(7):
                U[r+12][c+14] += u2[r][c]
                V[r+8][c+14] += v2[r][c]
                V[r][c+14] -= v2[r][c]
                W[r+8][c+14] += w2[r][c]

        # Term 5: A11+A21, B22
        # U: A11 (0-3) + A21 (8-11)
        # V: B22 (12-15)
        # W: C12 (4-7) - C11 (0-3)
        for r in range(4):
            for c in range(7):
                U[r][c+21] += u2[r][c]
                U[r+8][c+21] += u2[r][c]
                V[r+12][c+21] += v2[r][c]
                W[r+4][c+21] += w2[r][c]
                W[r][c+21] -= w2[r][c]

        # Term 6: A21-A11, B11+B12
        # U: A21 (8-11) - A11 (0-3)
        # V: B11 (0-3) + B12 (4-7)
        # W: C22 (12-15)
        for r in range(4):
            for c in range(7):
                U[r+8][c+28] += u2[r][c]
                U[r][c+28] -= u2[r][c]
                V[r][c+28] += v2[r][c]
                V[r+4][c+28] += v2[r][c]
                W[r+12][c+28] += w2[r][c]

        # Term 7: A12-A22, B21+B22
        # U: A12 (4-7) - A22 (12-15)
        # V: B21 (8-11) + B22 (12-15)
        # W: C11 (0-3)
        for r in range(4):
            for c in range(7):
                U[r+4][c+35] += u2[r][c]
                U[r+12][c+35] -= u2[r][c]
                V[r+8][c+35] += v2[r][c]
                V[r+12][c+35] += v2[r][c]
                W[r][c+35] += w2[r][c]

        return U, V, W

    # Schoolbook for 4x4x1 (rank 16)
    def schoolbook_4x4x1():
        U = [[0]*16 for _ in range(16)]
        V = [[0]*16 for _ in range(4)]
        W = [[0]*16 for _ in range(4)]
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Construct 4x4x9
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Part 1: B1 (cols 0-3)
    u1, v1, w1 = strassen_4x4()
    for r in range(16):
        for c in range(49):
            U[r][c] += u1[r][c]
            V[r][c] += v1[r][c]
            W[r][c] += w1[r][c]
            
    # Part 2: B2 (cols 4-7)
    u2, v2, w2 = strassen_4x4()
    for r in range(16):
        for c in range(49):
            U[r][c+49] += u2[r][c]
            V[r+4][c+49] += v2[r][c]
            W[r+4][c+49] += w2[r][c]
            
    # Part 3: B3 (col 8)
    u3, v3, w3 = schoolbook_4x4x1()
    for r in range(16):
        for c in range(16):
            U[r][c+98] += u3[r][c]
            V[r+8][c+98] += v3[r][c]
            W[r+8][c+98] += w3[r][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
