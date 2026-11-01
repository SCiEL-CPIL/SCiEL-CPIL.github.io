"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition: 7 (Strassen for 2x2) + 7 (Strassen for 2x2) + 4 (Schoolbook for 2x1)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(block_start_k, col_offset):
        # B indices:
        # B_00 = B[0, block_start_k] -> V[0 * p + block_start_k]
        # B_01 = B[0, block_start_k + 1] -> V[0 * p + block_start_k + 1]
        # B_10 = B[1, block_start_k] -> V[1 * p + block_start_k]
        # B_11 = B[1, block_start_k + 1] -> V[1 * p + block_start_k + 1]
        
        v_00 = 0 * p + block_start_k
        v_01 = 0 * p + block_start_k + 1
        v_10 = 1 * p + block_start_k
        v_11 = 1 * p + block_start_k + 1
        
        # W indices:
        # C_00 = C[0, block_start_k] -> W[block_start_k * n + 0]
        # C_01 = C[0, block_start_k + 1] -> W[(block_start_k + 1) * n + 0]
        # C_10 = C[1, block_start_k] -> W[block_start_k * n + 1]
        # C_11 = C[1, block_start_k + 1] -> W[(block_start_k + 1) * n + 1]
        
        w_00 = block_start_k * n + 0
        w_01 = (block_start_k + 1) * n + 0
        w_10 = block_start_k * n + 1
        w_11 = (block_start_k + 1) * n + 1
        
        # U indices:
        # A_00 = U[0]
        # A_01 = U[1]
        # A_10 = U[2]
        # A_11 = U[3]
        
        # M1: (A00 + A11) * (B00 + B11)
        U[0][col_offset] = 1
        U[3][col_offset] = 1
        V[v_00][col_offset] = 1
        V[v_11][col_offset] = 1
        W[w_00][col_offset] = 1
        W[w_11][col_offset] = 1
        
        # M2: (A10 + A11) * B00
        U[2][col_offset + 1] = 1
        U[3][col_offset + 1] = 1
        V[v_00][col_offset + 1] = 1
        W[w_01][col_offset + 1] = 1
        W[w_11][col_offset + 1] = -1
        
        # M3: A00 * (B10 - B11)
        U[0][col_offset + 2] = 1
        V[v_10][col_offset + 2] = 1
        V[v_11][col_offset + 2] = -1
        W[w_10][col_offset + 2] = 1
        W[w_11][col_offset + 2] = 1
        
        # M4: A11 * (B01 - B00)
        U[3][col_offset + 3] = 1
        V[v_01][col_offset + 3] = 1
        V[v_00][col_offset + 3] = -1
        W[w_00][col_offset + 3] = 1
        W[w_11][col_offset + 3] = 1
        
        # M5: (A00 + A01) * B11
        U[0][col_offset + 4] = 1
        U[1][col_offset + 4] = 1
        V[v_11][col_offset + 4] = 1
        W[w_00][col_offset + 4] = -1
        W[w_01][col_offset + 4] = 1
        
        # M6: (A10 - A00) * (B00 + B01)
        U[2][col_offset + 5] = 1
        U[0][col_offset + 5] = -1
        V[v_00][col_offset + 5] = 1
        V[v_01][col_offset + 5] = 1
        W[w_10][col_offset + 5] = 1
        W[w_11][col_offset + 5] = 1
        
        # M7: (A01 - A11) * (B10 + B11)
        U[1][col_offset + 6] = 1
        U[3][col_offset + 6] = -1
        V[v_10][col_offset + 6] = 1
        V[v_11][col_offset + 6] = 1
        W[w_00][col_offset + 6] = 1
        W[w_10][col_offset + 6] = 1

    def add_schoolbook(k_idx, col_offset):
        # B_0 = B[0, k_idx] -> V[0 * p + k_idx]
        # B_1 = B[1, k_idx] -> V[1 * p + k_idx]
        
        v_0 = 0 * p + k_idx
        v_1 = 1 * p + k_idx
        
        # C_0 = C[0, k_idx] -> W[k_idx * n + 0]
        # C_1 = C[1, k_idx] -> W[k_idx * n + 1]
        
        w_0 = k_idx * n + 0
        w_1 = k_idx * n + 1
        
        # M: A00 * B0
        U[0][col_offset] = 1
        V[v_0][col_offset] = 1
        W[w_0][col_offset] = 1
        
        # M: A01 * B1
        U[1][col_offset + 1] = 1
        V[v_1][col_offset + 1] = 1
        W[w_0][col_offset + 1] = 1
        
        # M: A10 * B0
        U[2][col_offset + 2] = 1
        V[v_0][col_offset + 2] = 1
        W[w_1][col_offset + 2] = 1
        
        # M: A11 * B1
        U[3][col_offset + 3] = 1
        V[v_1][col_offset + 3] = 1
        W[w_1][col_offset + 3] = 1

    # Block 1: k=0,1 (Strassen)
    add_strassen(0, 0)
    
    # Block 2: k=2,3 (Strassen)
    add_strassen(2, 7)
    
    # Block 3: k=4 (Schoolbook)
    add_schoolbook(4, 14)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
