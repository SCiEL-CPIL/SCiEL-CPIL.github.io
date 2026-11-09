"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen matrices
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Helper to get block and local indices from global index
    def get_indices(idx):
        # idx = i_block_row * 8 + i_local_row * 4 + i_block_col * 2 + i_local_col
        i_local_col = idx % 2
        i_block_col = (idx // 2) % 2
        i_local_row = (idx // 4) % 2
        i_block_row = idx // 8
        i_block = i_block_row * 2 + i_block_col
        i_local = i_local_row * 2 + i_local_col
        return i_block, i_local
        
    # Construct U, V, W
    for idx in range(16):
        i_block, i_local = get_indices(idx)
        for k in range(7):
            for l in range(7):
                c = k * 7 + l
                U[idx][c] = U2[i_block][k] * U2[i_local][l]
                V[idx][c] = V2[i_block][k] * V2[i_local][l]
                W[idx][c] = W2[i_block][k] * W2[i_local][l]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
