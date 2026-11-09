"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # We decompose 4x4x6 into 4x4x4 and 4x4x2
    # Part 1: 4x4x4 (columns 0-3 of C)
    # Part 2: 4x4x2 (columns 4-5 of C)
    
    # Helper to add a 2x2x2 block at offset (bi, bj, bk) with rank offset col_start
    def add_block(bi, bj, bk, col_start):
        for c in range(7):
            col = col_start + c
            for di in range(2):
                for dj in range(2):
                    i = bi * 2 + di
                    j = bj * 2 + dj
                    U[i * m + j][col] = S_U[di * 2 + dj][c]
            for dj in range(2):
                for dk in range(2):
                    j = bj * 2 + dj
                    k = bk * 2 + dk
                    V[j * p + k][col] = S_V[dj * 2 + dk][c]
            for dk in range(2):
                for di in range(2):
                    k = bk * 2 + dk
                    i = bi * 2 + di
                    W[k * n + i][col] = S_W[dk * 2 + di][c]

    # Total rank:
    # 4x4x4: 2x2x2 blocks -> 2*2*2 = 8 blocks * 7 = 56
    # 4x4x2: 2x2x1 blocks? No, 4x4x2 is 2x2 blocks in i,j and 1 block in k (size 2).
    # i: 2 blocks, j: 2 blocks, k: 1 block (size 2).
    # 2*2*1 = 4 blocks * 7 = 28.
    # Total = 56 + 28 = 84.
    
    total_rank = 84
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Part 1: 4x4x4
    # k ranges 0..3, so bk in 0,1
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                add_block(bi, bj, bk, col_offset)
                col_offset += 7
                
    # Part 2: 4x4x2
    # k ranges 4..5, so bk=2 (offset 4)
    # But our block logic assumes k starts at bk*2.
    # For bk=2, k is 4,5.
    for bi in range(2):
        for bj in range(2):
            bk = 2 # k offset 4
            add_block(bi, bj, bk, col_offset)
            col_offset += 7
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
