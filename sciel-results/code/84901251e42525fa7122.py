import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 for 4x4x4
    rank_444 = 49
    # Rank 16 for 4x4x1
    rank_441 = 16
    
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Decomposition (Strassen) ---
    # We need to generate the 49 terms for M_4,4,4.
    # We use a recursive Strassen generator.
    
    def get_strassen_2x2():
        # Returns list of 7 tuples (u, v, w) for 2x2 matrices
        # u, v, w are lists of 4 floats
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]

    def expand_4x4_strassen():
        # Generates 49 terms for 4x4 matrices
        # Each term is (u, v, w) of length 16
        base_patterns = get_strassen_2x2()
        terms = []
        
        # Outer Strassen: 7 combinations of 2x2 blocks
        # A' is 2x2 matrix of 2x2 blocks.
        # B' is 2x2 matrix of 2x2 blocks.
        # M_k = A'_k B'_k (conceptually, but A'_k is a matrix)
        
        # Actually, we apply Strassen to the blocks.
        # For each outer term k (0..6):
        #   A_k is a 2x2 matrix (blocks of A)
        #   B_k is a 2x2 matrix (blocks of B)
        #   We compute C_k = A_k B_k using inner Strassen.
        #   C_k is a 2x2 matrix (blocks of C)
        #   Then C is formed by combining C_k.
        
        # Let's trace the coefficients.
        # Outer u_k, v_k are 2x2 matrices of coefficients for blocks.
        # But the "scalar" in outer Strassen is a 2x2 matrix.
        # So we need to expand the multiplication of two 2x2 matrices (A_k, B_k) into 7 scalar products.
        
        # We can do this by treating A_k and B_k as 2x2 matrices of scalars (where each scalar is a block).
        # Wait, A_k is a 2x2 matrix where each entry is a 2x2 block of A.
        # So A_k is effectively a 4x4 matrix with a specific sparsity/pattern.
        
        # Let's generate the 4x4 Strassen terms directly.
        # There are 7 outer terms. For each, we have 7 inner terms.
        
        # Outer patterns (for blocks):
        # u_block[k][block_idx] = coeff
        # v_block[k][block_idx] = coeff
        
        # Block indices: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        outer_u = [
            [1, 0, 0, 1], # 0
            [0, 0, 1, 1], # 1
            [1, 0, 0, 0], # 2
            [0, 0, 0, 1], # 3
            [1, 1, 0, 0], # 4
            [-1, 0, 1, 0],# 5
            [0, 1, 0, -1] # 6
        ]
        outer_v = [
            [1, 0, 0, 1], # 0
            [1, 0, 0, 0], # 1
            [0, 1, 0, -1],# 2
            [0, 0, 1, -1],# 3
            [0, 0, 0, 1], # 4
            [1, 1, 0, 0], # 5
            [0, 0, 1, 1]  # 6
        ]
        
        # Inner patterns (for elements within blocks)
        inner_patterns = get_strassen_2x2()
        
        # Outer W patterns: how M_k contributes to C blocks
        # C_00 = M_0 + M_3 - M_4 + M_6
        # C_01 = M_2 + M_4
        # C_10 = M_1 + M_3
        # C_11 = M_0 - M_1 + M_2 + M_5
        
        outer_w = [
            [1, 0, 0, 1], # 0
            [0, 0, 1, -1],# 1
            [0, 1, 0, 1], # 2
            [1, 0, 1, 0], # 3
            [-1, 1, 0, 0],# 4
            [0, 0, 0, 1], # 5
            [1, 0, 0, 0]  # 6
        ]
        
        for k in range(7):
            # A_k is formed by outer_u[k] * Blocks_A
            # B_k is formed by outer_v[k] * Blocks_B
            # We need to compute C_k = A_k B_k.
            # A_k and B_k are 2x2 matrices.
            # We use inner Strassen to decompose C_k.
            
            # For each inner term l (0..6):
            for l in range(7):
                u_in, v_in, w_in = inner_patterns[l]
                
                # u_in is coeffs for the 4 elements of A_k (which are blocks)
                # v_in is coeffs for the 4 elements of B_k (which are blocks)
                # w_in is coeffs for the 4 elements of C_k (which are blocks)
                
                # Construct u (length 16)
                u = [0] * 16
                v = [0] * 16
                w = [0] * 16
                
                # u and v
                for block_idx in range(4):
                    coeff_a = outer_u[k][block_idx]
                    coeff_b = outer_v[k][block_idx]
                    
                    # A_k[block_idx] = coeff_a * A_block[block_idx]
                    # The inner u_in[block_idx] applies to A_k[block_idx]
                    # So the coefficient for A_block[block_idx] is coeff_a * u_in[block_idx]
                    
                    # Map block_idx and inner element to 16-dim index
                    # Block 0: rows 0-1, cols 0-1
                    # Block 1: rows 0-1, cols 2-3
                    # Block 2: rows 2-3, cols 0-1
                    # Block 3: rows 2-3, cols 2-3
                    
                    base_a = block_idx * 4
                    base_b = block_idx * 4
                    
                    for elem in range(4):
                        u[base_a + elem] += coeff_a * u_in[elem]
                        v[base_b + elem] += coeff_b * v_in[elem]
                
                # w
                # w_in[block_idx] is the coeff for C_k[block_idx]
                # C_k[block_idx] contributes to C_block[block_idx] with coeff outer_w[k][block_idx]
                for block_idx in range(4):
                    coeff_c = outer_w[k][block_idx]
                    if coeff_c != 0:
                        base_c = block_idx * 4
                        # w_in[block_idx] is a scalar coeff for the block
                        # But wait, w_in is for the 2x2 matrix C_k.
                        # C_k is a 2x2 matrix of blocks.
                        # The inner Strassen w_in gives the coeff for each of the 4 blocks of C_k.
                        # So C_k[block_idx] = w_in[block_idx] * P_{k,l}
                        # And C_block[block_idx] += outer_w[k][block_idx] * C_k[block_idx]
                        # So C_block[block_idx] += outer_w[k][block_idx] * w_in[block_idx] * P_{k,l}
                        
                        # The entries of C_block[block_idx] are all multiplied by this scalar?
                        # No, P_{k,l} is a scalar.
                        # C_k[block_idx] is a 2x2 block.
                        # In the inner Strassen, w_in[elem] is the coeff for C_k_elem (scalar).
                        # Here C_k_elem is a 2x2 block.
                        # So the entire block C_k[block_idx] is scaled by w_in[block_idx].
                        # And then that block is added to C_block[block_idx] scaled by outer_w[k][block_idx].
                        
                        # So for all 4 elements in C_block[block_idx], the coeff is outer_w[k][block_idx] * w_in[block_idx].
                        
                        val = outer_w[k][block_idx] * w_in[block_idx]
                        for elem in range(4):
                            w[base_c + elem] += val

                terms.append((u, v, w))
                
        return terms

    terms_444 = expand_4x4_strassen()
    
    for idx, (u, v, w) in enumerate(terms_444):
        col = idx
        # U
        for i in range(16):
            U[i][col] = u[i]
        # V
        # V shape 20x49. Rows 0-15 are for B cols 0-3.
        for i in range(16):
            V[i][col] = v[i]
        # Rows 16-19 are 0
        # W
        # W shape 20x49. Rows 0-15 are for C cols 0-3.
        for i in range(16):
            W[i][col] = w[i]
        # Rows 16-19 are 0

    # --- Part 2: 4x4x1 Decomposition ---
    # Rank 16
    # A is 4x4, B is 4x1 (col 4 of B), C is 4x1 (col 4 of C)
    for t in range(16):
        col = rank_444 + t
        i = t // 4
        j = t % 4
        
        # U[i*4+j] = 1
        U[i * 4 + j][col] = 1
        
        # V[j*5+4] = 1 (B is 4x5, we use col 4)
        V[j * 5 + 4][col] = 1
        
        # W[4*4+i] = 1 (C is 4x5, we use col 4, index k=4)
        W[4 * 4 + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
