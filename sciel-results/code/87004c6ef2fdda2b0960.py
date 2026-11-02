"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block Strassen decomposition for 6x6 matrix multiplication.
    # Use 2x2 blocks in a 3x3 block structure.
    # Each block multiplication is 2x2 x 2x2, which has rank 7 (Strassen).
    # Total rank = 27 * 7 = 189.
    
    n, m, p = DIMENSIONS
    block_size = 2
    num_blocks = n // block_size  # 3
    
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen terms for 2x2 matrix multiplication
    # Each term is (u_vec, v_vec, w_vec) where each vector has 4 components
    # Indexing for 2x2 blocks: [0,0], [0,1], [1,0], [1,1]
    strassen_terms = [
        # (U_factors, V_factors, W_factors)
        ([1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]),
        ([1, 0, 1, 0], [0, 1, 0, -1], [0, 1, 1, 0]),
        ([0, 1, 0, -1], [1, 1, 0, 0], [0, 0, 1, 0]),
        ([0, -1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]),
    ]
    
    col = 0
    for I in range(num_blocks):  # Block row of A
        for K in range(num_blocks):  # Block col of A = Block row of B
            for J in range(num_blocks):  # Block col of B
                for t in range(7):
                    u_vec, v_vec, w_vec = strassen_terms[t]
                    
                    # For each non-zero component in the Strassen factors
                    for idx in range(4):
                        if u_vec[idx] != 0:
                            # Map block indices to global indices
                            i = I * block_size + (idx % 2)
                            j = K * block_size + (idx // 2)
                            U[i * m + j][col] += u_vec[idx]
                        
                        if v_vec[idx] != 0:
                            # B block: row K, col J
                            r = K * block_size + (idx // 2)
                            c = J * block_size + (idx % 2)
                            V[r * p + c][col] += v_vec[idx]
                        
                        if w_vec[idx] != 0:
                            # C block: row I, col J
                            k = J * block_size + (idx % 2)
                            o = I * block_size + (idx // 2)
                            W[k * n + o][col] += w_vec[idx]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

