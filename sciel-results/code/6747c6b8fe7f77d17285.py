import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition vectors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # c=1: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    # c=2: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    # c=3: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    # c=4: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    # c=5: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # c=6: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Define the 7 outer Strassen terms for 4x4x4
    # Each term is a combination of 2x2 blocks
    # We represent the combination of blocks for U, V, W
    # Block indices: 00, 01, 10, 11
    # We need to map the 2x2 Strassen vectors to 4x4 vectors
    
    # Helper to expand a 2x2 Strassen vector to 4x4 based on block placement
    def expand_2x2_to_4x4(vec_2x2, block_rows, block_cols):
        """
        vec_2x2: shape (4, 1) or similar, representing the 2x2 tensor factor
        block_rows: tuple of 2 row indices for the 2x2 blocks
        block_cols: tuple of 2 col indices for the 2x2 blocks
        Returns a 4x4 matrix (flattened to 16)
        """
        res = np.zeros(16)
        for r in range(2):
            for c in range(2):
                val = vec_2x2[r * 2 + c]
                if val != 0:
                    gr = block_rows[r] * 2
                    gc = block_cols[c] * 2
                    # Fill the 2x2 subblock
                    res[(gr + 0) * 4 + (gc + 0)] += val
                    res[(gr + 0) * 4 + (gc + 1)] += 0 # Wait, vec_2x2 is for the whole 2x2 multiplication?
                    # No, vec_2x2 is the factor for the 2x2 multiplication.
                    # The 2x2 multiplication is $C_{ab} = \sum_{cd} A_{ac} B_{cd}$.
                    # The tensor factor $U$ for $2 \times 2$ has indices $a, c$.
                    # When we place this into a $4 \times 4$ block at row $R$, col $C$,
                    # the indices become $4R+a, 4C+c$.
                    
                    # Let's re-verify the index mapping.
                    # In 2x2 case:
                    # U index: a * 2 + c. (a=row, c=col of A)
                    # V index: c * 2 + d. (c=row of B, d=col of B)
                    # W index: d * 2 + a. (d=row of C? No, C is 2x2. d is col of C, a is row of C?
                    # Prompt: W[k*n+i, r]. For 2x2, n=2. W[2*d+a, r]?
                    # In seed code for 2x2:
                    # global_w_idx = global_k * 4 + global_i.
                    # Here global_k is col of C (d), global_i is row of C (a).
                    # So W index is d * 2 + a.
                    
                    # So for 4x4:
                    # U index: (4*R + a) * 4 + (4*C + c) = 16R + 4a + 4C + c.
                    # V index: (4*C + c) * 8 + (4*Col + d) = 32C + 8c + 4Col + d.
                    # W index: (4*Col + d) * 4 + (4*R + a) = 16Col + 4d + 4R + a.
                    
                    # My vec_2x2 is already flattened.
                    # vec_2x2[0] corresponds to a=0, c=0.
                    # vec_2x2[1] corresponds to a=0, c=1.
                    # vec_2x2[2] corresponds to a=1, c=0.
                    # vec_2x2[3] corresponds to a=1, c=1.
                    
                    # So I just need to place the 4 values of vec_2x2 into the correct positions in the 16-length vector.
                    pass
        return res

    # This helper is getting complicated. Let's just do it explicitly in the loop.
    
    # Outer Strassen Terms
    # 0: (A00+A11)(B00+B11)
    # 1: (A10+A11)B00
    # 2: A00(B01-B11)
    # 3: A11(B10-B00)
    # 4: (A00+A01)B11
    # 5: (A10-A00)(B00+B01)
    # 6: (A01-A11)(B10+B11)
    
    # For each outer term k_out, we have:
    # U blocks: list of (coeff, row_block, col_block)
    # V blocks: list of (coeff, row_block, col_block)
    # W blocks: list of (coeff, row_block, col_block)
    
    outer_specs = [
        # k_out 0
        {
            'U': [(1, 0, 0), (1, 1, 1)], # A00 + A11
            'V': [(1, 0, 0), (1, 1, 1)], # B00 + B11
            'W': [(1, 0, 0), (1, 1, 1)]  # C00 + C11
        },
        # k_out 1
        {
            'U': [(1, 1, 0), (1, 1, 1)], # A10 + A11
            'V': [(1, 0, 0)],            # B00
            'W': [(1, 1, 0), (-1, 1, 1)] # C10 - C11
        },
        # k_out 2
        {
            'U': [(1, 0, 0)],            # A00
            'V': [(1, 0, 1), (-1, 1, 1)],# B01 - B11
            'W': [(1, 0, 1), (1, 1, 1)]  # C01 + C11
        },
        # k_out 3
        {
            'U': [(1, 1, 1)],            # A11
            'V': [(1, 1, 0), (-1, 0, 0)],# B10 - B00
            'W': [(1, 0, 0), (1, 1, 0)]  # C00 + C10
        },
        # k_out 4
        {
            'U': [(1, 0, 0), (1, 0, 1)], # A00 + A01
            'V': [(1, 1, 1)],            # B11
            'W': [(-1, 0, 0), (1, 0, 1)] # -C00 + C01
        },
        # k_out 5
        {
            'U': [(1, 1, 0), (-1, 0, 0)],# A10 - A00
            'V': [(1, 0, 0), (1, 0, 1)], # B00 + B01
            'W': [(1, 1, 1)]             # C11
        },
        # k_out 6
        {
            'U': [(1, 0, 1), (-1, 1, 1)],# A01 - A11
            'V': [(1, 1, 0), (1, 1, 1)], # B10 + B11
            'W': [(1, 0, 0)]             # C00
        }
    ]

    for half in range(2):
        for k_out in range(7):
            for k_in in range(7):
                col = half * 49 + k_out * 7 + k_in
                
                # U
                for coeff_u, rb, cb in outer_specs[k_out]['U']:
                    u_vec = coeff_u * U2[:, k_in]
                    for r in range(2):
                        for c in range(2):
                            val = u_vec[r * 2 + c]
                            if val != 0:
                                gr = rb * 2 + r
                                gc = cb * 2 + c
                                u_idx = gr * 4 + gc
                                U[u_idx, col] += val

                # V
                for coeff_v, rb, cb in outer_specs[k_out]['V']:
                    v_vec = coeff_v * V2[:, k_in]
                    for r in range(2):
                        for c in range(2):
                            val = v_vec[r * 2 + c]
                            if val != 0:
                                gr = rb * 2 + r
                                gc = cb * 2 + c
                                # V index: row of B (gr), col of B (gc + half*4)
                                v_idx = gr * 8 + (gc + half * 4)
                                V[v_idx, col] += val

                # W
                for coeff_w, rb, cb in outer_specs[k_out]['W']:
                    w_vec = coeff_w * W2[:, k_in]
                    for r in range(2):
                        for c in range(2):
                            val = w_vec[r * 2 + c]
                            if val != 0:
                                gr = rb * 2 + r
                                gc = cb * 2 + c
                                # W index: col of C (gc + half*4), row of C (gr)
                                # Prompt: W[k*n+i, r] -> index = (col) * 4 + (row)
                                w_idx = (gc + half * 4) * 4 + gr
                                W[w_idx, col] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
