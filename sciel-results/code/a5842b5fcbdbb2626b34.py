"""Matrix multiplication 4x4x8 via Strassen recursion."""

def construct():
    n, m, p = 4, 4, 8
    
    # Rank-7 decomposition for 2x2x2 (Strassen)
    # U, V, W are 7 vectors of length 4
    U_2 = [
        [1, 0, 0, 0],  # M1: A00
        [1, 1, 0, 0],  # M2: A00+A01
        [0, 0, 1, 1],  # M3: A10+A11
        [0, 0, 0, 1],  # M4: A11
        [1, 0, 1, 0],  # M5: A00+A10
        [0, 1, 0, -1], # M6: A01-A11
        [1, -1, 0, 0]  # M7: A00-A01
    ]
    V_2 = [
        [1, 1, 0, 0],  # M1: B00+B01
        [0, 0, 0, 1],  # M2: B11
        [1, 0, 0, 0],  # M3: B00
        [0, 0, 1, 1],  # M4: B10+B11
        [0, 1, 0, 1],  # M5: B01+B11
        [0, 0, 1, 1],  # M6: B10+B11
        [1, 0, 1, 0]   # M7: B00+B10
    ]
    W_2 = [
        [1, 0, 0, 1],  # M1: C00+C11
        [0, 1, 0, 0],  # M2: C01
        [0, 0, 1, -1], # M3: C10-C11
        [-1, 1, 0, 0], # M4: -C00+C01
        [0, 0, 1, 0],  # M5: C10
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]
    
    # Meta-vectors for 4x4x4 (Strassen on blocks)
    # Each is a 4x4 matrix (flat 16)
    # Blocks: 00, 01, 10, 11
    # U_meta[i] is the linear form on A for term i
    # V_meta[i] is the linear form on B for term i
    # W_meta[i] is the linear form on C for term i
    
    def block_to_flat(block, val):
        # block is (r, c) in 2x2 grid of 2x2 blocks
        # val is scalar
        # Returns 16-length vector
        vec = [0] * 16
        r, c = block
        base_i = r * 2
        base_j = c * 2
        for i in range(2):
            for j in range(2):
                idx = (base_i + i) * 4 + (base_j + j)
                vec[idx] = val
        return vec

    def add_block(vec, block, val):
        # Add val * Identity to block (r,c) in vec
        r, c = block
        base_i = r * 2
        base_j = c * 2
        for i in range(2):
            for j in range(2):
                idx = (base_i + i) * 4 + (base_j + j)
                vec[idx] += val
        return vec

    U_meta = [[0]*16 for _ in range(7)]
    V_meta = [[0]*16 for _ in range(7)]
    W_meta = [[0]*16 for _ in range(7)]
    
    # M1: A00 * (B00 + B01)
    add_block(U_meta[0], (0,0), 1)
    add_block(V_meta[0], (0,0), 1)
    add_block(V_meta[0], (0,1), 1)
    add_block(W_meta[0], (0,0), 1)
    add_block(W_meta[0], (1,1), 1)
    
    # M2: (A00 + A01) * B11
    add_block(U_meta[1], (0,0), 1)
    add_block(U_meta[1], (0,1), 1)
    add_block(V_meta[1], (1,1), 1)
    add_block(W_meta[1], (0,1), 1)
    
    # M3: (A10 + A11) * B00
    add_block(U_meta[2], (1,0), 1)
    add_block(U_meta[2], (1,1), 1)
    add_block(V_meta[2], (0,0), 1)
    add_block(W_meta[2], (1,0), 1)
    add_block(W_meta[2], (1,1), -1)
    
    # M4: A11 * (B10 + B11)
    add_block(U_meta[3], (1,1), 1)
    add_block(V_meta[3], (1,0), 1)
    add_block(V_meta[3], (1,1), 1)
    add_block(W_meta[3], (0,0), -1)
    add_block(W_meta[3], (0,1), 1)
    
    # M5: (A00 + A10) * (B01 + B11)
    add_block(U_meta[4], (0,0), 1)
    add_block(U_meta[4], (1,0), 1)
    add_block(V_meta[4], (0,1), 1)
    add_block(V_meta[4], (1,1), 1)
    add_block(W_meta[4], (1,0), 1)
    
    # M6: (A01 - A11) * (B10 + B11)
    add_block(U_meta[5], (0,1), 1)
    add_block(U_meta[5], (1,1), -1)
    add_block(V_meta[5], (1,0), 1)
    add_block(V_meta[5], (1,1), 1)
    add_block(W_meta[5], (1,1), 1)
    
    # M7: (A00 - A01) * (B00 + B10)
    add_block(U_meta[6], (0,0), 1)
    add_block(U_meta[6], (0,1), -1)
    add_block(V_meta[6], (0,0), 1)
    add_block(V_meta[6], (1,0), 1)
    add_block(W_meta[6], (0,0), 1)

    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(7):
        for J in range(7):
            # Construct U_4, V_4, W_4 for this term
            # U_4: For each block in U_meta[I], add val * U_2[J]
            U_4 = [0] * 16
            V_4 = [0] * 16
            W_4 = [0] * 16
            
            # We need to map the 16-length meta vectors to the 16-length base vectors
            # Actually, the meta vector already has the block structure.
            # U_meta[I] is 16-length.
            # U_2[J] is 4-length.
            # The product is not a simple dot product.
            # The rank-1 tensor is formed by:
            # U_total[a] = sum_{blocks} U_meta[I][block_pos] * U_2[J][local_pos]
            # This is effectively an outer product of the block coefficients and local coefficients.
            
            # Let's just iterate through the 4 blocks.
            for br in range(2):
                for bc in range(2):
                    # Get the coefficient of this block in U_meta[I]
                    # The block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1
                    # In flat index:
                    # We can just take the first element of the block in U_meta[I]
                    # Since U_meta[I] is constant within a block (identity * val)
                    base_idx = (br * 2) * 4 + (bc * 2)
                    val_u = U_meta[I][base_idx]
                    val_v = V_meta[I][base_idx]
                    val_w = W_meta[I][base_idx]
                    
                    if val_u != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                flat_idx = (br * 2 + li) * 4 + (bc * 2 + lj)
                                U_4[flat_idx] = val_u * U_2[J][local_idx]
                                
                    if val_v != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                flat_idx = (br * 2 + li) * 4 + (bc * 2 + lj)
                                V_4[flat_idx] = val_v * V_2[J][local_idx]
                                
                    if val_w != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                flat_idx = (br * 2 + li) * 4 + (bc * 2 + lj)
                                W_4[flat_idx] = val_w * W_2[J][local_idx]

            # Assign to U, V, W for the first half (C_L)
            for i in range(16):
                U[i][col] = U_4[i]
                V[i][col] = V_4[i] # V_L is first 16 rows
                W[i][col] = W_4[i] # W_L is first 16 rows (k=0..3)
            
            # Assign to U, V, W for the second half (C_R)
            # V_R is rows 16..31
            # W_R is columns 4..7, so indices (k+4)*4 + i
            for i in range(16):
                U[i][col + 49] = U_4[i]
                V[i + 16][col + 49] = V_4[i]
                # W index mapping:
                # W_4[i] corresponds to C_L[k, i] where k = i // 4, i_col = i % 4
                # We need C_R[k, i] which is column k+4
                k = i // 4
                i_col = i % 4
                w_idx = (k + 4) * 4 + i_col
                W[w_idx][col + 49] = W_4[i]
                
            col += 1

    return {"U": U, "V": V, "W": W}
