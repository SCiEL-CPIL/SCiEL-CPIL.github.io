"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition for M_{2,4,8}
    # Split B (4x8) into B1 (4x4) and B2 (4x4)
    # Each M_{2,4,4} has rank 16
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to fill a M_{2,4,4} block
    def fill_block(offset_U, offset_V, offset_W, k_offset):
        # U: 8x16, V: 16x16, W: 8x16
        # V is identity
        for j in range(m):
            for k_local in range(4):
                k_global = k_offset + k_local
                v_row = j * p + k_global
                v_col = offset_V + j * 4 + k_local
                V[v_row][v_col] = 1
                
        # U and W
        for i in range(n):
            for j in range(m):
                for k_local in range(4):
                    k_global = k_offset + k_local
                    u_row = i * m + j
                    u_col = offset_U + j * 4 + k_local
                    U[u_row][u_col] = 1
                    
                    w_row = k_global * n + i
                    w_col = offset_W + j * 4 + k_local
                    W[w_row][w_col] = 1

    # Block 1: k in 0..3
    fill_block(0, 0, 0, 0)
    # Block 2: k in 4..7
    fill_block(16, 16, 16, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
