"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # 2x2x2 Strassen rank-7 decomposition
    # U: 4x7, rows (i,j) for i,j in {0,1}
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    # V: 4x7, rows (j,k) for j,k in {0,1}
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    # W: 4x7, rows (k,i) for k,i in {0,1}
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]
    
    # Split into blocks: i:2x2, j:2x2, k:3x2
    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2
    
    total_rank = n_blocks * m_blocks * p_blocks * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                # For each of the 7 Strassen products
                for c in range(7):
                    col = col_offset + c
                    # U: rows (i,j) for i in [bi*2, bi*2+2), j in [bj*2, bj*2+2)
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    
                    # V: rows (j,k) for j in [bj*2, bj*2+2), k in [bk*2, bk*2+2)
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    
                    # W: rows (k,i) for k in [bk*2, bk*2+2), i in [bi*2, bi*2+2)
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                
                col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
