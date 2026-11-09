import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank calculation:
    # Split B into B1 (4x4) and B2 (4x2).
    # C = A*B = [A*B1, A*B2]
    # Rank(A*B1) for 4x4x4x4 is 49 (Strassen).
    # Rank(A*B2) for 4x4x4x2 is 28 (4 Strassen blocks of 2x2x2x2).
    # Total Rank = 49 + 28 = 77.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2x2x2
    # U: A part, V: B part, W: C part
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    alpha = np.array([
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ])
    beta = np.array([
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [-1, 0, 1, 0],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ])
    # Reconstruction: W = sum gamma * M_k
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # recon[c_row, c_col, k] = gamma
    recon = np.zeros((4, 7))
    # c_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    recon[0, 0] = 1 # M1
    recon[0, 3] = 1 # M4
    recon[0, 4] = -1 # M5
    recon[0, 6] = 1 # M7
    
    recon[1, 2] = 1 # M3
    recon[1, 4] = 1 # M5
    
    recon[2, 1] = 1 # M2
    recon[2, 3] = 1 # M4
    
    recon[3, 0] = 1 # M1
    recon[3, 1] = -1 # M2
    recon[3, 2] = 1 # M3
    recon[3, 5] = 1 # M6
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block to U, V, W
    # a_offsets: (row_offset, col_offset) for A block in global A (4x4)
    # b_offsets: (row_offset, col_offset) for B block in global B (4x6)
    # c_offsets: (row_offset, col_offset) for C block in global C (4x6)
    # Note: A is 4x4, B is 4x6, C is 4x6.
    # Global indices:
    # A: row 0-3, col 0-3.
    # B: row 0-3, col 0-5.
    # C: row 0-3, col 0-5.
    
    def add_block(a_r, a_c, b_r, b_c, c_r, c_c, sign=1):
        nonlocal col
        for k in range(7):
            # U
            for i in range(4):
                if alpha[k, i] != 0:
                    # i: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    ri, ci = i // 2, i % 2
                    u_idx = (a_r + ri) * 4 + (a_c + ci)
                    U[u_idx, col] += sign * alpha[k, i]
            
            # V
            for i in range(4):
                if beta[k, i] != 0:
                    ri, ci = i // 2, i % 2
                    v_idx = (b_r + ri) * 6 + (b_c + ci)
                    V[v_idx, col] += sign * beta[k, i]
            
            # W
            for i in range(4):
                if recon[i, k] != 0:
                    ri, ci = i // 2, i % 2
                    w_idx = (c_r + ri) * 6 + (c_c + ci)
                    W[w_idx, col] += sign * recon[i, k]
                    
            col += 1
    
    # Part 1: A * B1 (4x4 * 4x4)
    # B1 is cols 0-3 of B.
    # We use Strassen on 4x4 blocks.
    # Top-level Strassen terms (7 of them)
    # Each term involves a 2x2 matrix from A and a 2x2 matrix from B.
    # The 2x2 matrices are formed by linear combinations of the 2x2 blocks of A and B.
    # Let A blocks be A00, A01, A10, A11.
    # Let B1 blocks be B00, B01, B10, B11.
    
    # We need to decompose the product of two 2x2 matrices (formed by sums/diffs) into 7 scalar-like (actually 2x2 matrix) products?
    # No, the standard 4x4 Strassen works by:
    # 1. Form 7 linear combinations of A blocks (each is a 2x2 matrix).
    # 2. Form 7 linear combinations of B blocks (each is a 2x2 matrix).
    # 3. Multiply these 7 pairs of 2x2 matrices.
    # 4. Each 2x2 multiplication is decomposed into 7 terms.
    
    # So we iterate over the 7 top-level Strassen terms.
    # For each top-level term t, we have:
    # A_t = sum_{i} alpha_top[t, i] * A_block_i
    # B_t = sum_{i} beta_top[t, i] * B_block_i
    # Then we compute A_t * B_t using the 2x2 Strassen decomposition.
    
    # The alpha/beta/recon defined above are for the *inner* 2x2 multiplication.
    # We need the *outer* Strassen coefficients.
    # Outer Strassen is identical structure.
    # A_top = [[A00, A01], [A10, A11]]
    # B_top = [[B00, B01], [B10, B11]]
    
    # Let's define the outer coefficients. They are the same as alpha/beta/recon.
    # Outer term k_outer:
    # A_comb = sum_{j} alpha[k_outer, j] * A_block_j
    # B_comb = sum_{j} beta[k_outer, j] * B_block_j
    # Then C_comb = A_comb * B_comb.
    # C_comb is a 2x2 matrix.
    # We decompose C_comb = A_comb * B_comb into 7 inner terms.
    # For each inner term k_inner:
    #   A_inner = sum_{i} alpha[k_inner, i] * (element of A_comb)
    #   B_inner = sum_{i} beta[k_inner, i] * (element of B_comb)
    #   C_inner = sum_{i} recon[i, k_inner] * A_inner * B_inner
    
    # This is getting complex to nest.
    # Alternative: Just expand the 49 terms directly.
    # Term (k_outer, k_inner).
    # U contribution:
    #   From A_comb: sum_j alpha[k_outer, j] * A_block_j.
    #   From A_inner: sum_i alpha[k_inner, i] * (element i of A_comb).
    #   So U = sum_j sum_i alpha[k_outer, j] * alpha[k_inner, i] * (element i of A_block_j).
    #   Wait, A_block_j is a 2x2 matrix. "element i" refers to the position in the 2x2 matrix.
    #   Let A_block_j = [[A_j00, A_j01], [A_j10, A_j11]].
    #   The 2x2 Strassen acts on the 2x2 matrix A_comb.
    #   A_comb = [[A_comb00, A_comb01], [A_comb10, A_comb11]].
    #   A_comb_rc = sum_j alpha[k_outer, j] * A_block_j_rc.
    #   The inner Strassen term k_inner uses A_comb elements.
    #   U_inner_i = alpha[k_inner, i] * A_comb_{pos(i)}.
    #   So U_total = sum_j sum_i alpha[k_outer, j] * alpha[k_inner, i] * A_block_j_{pos(i)}.
    
    # Let's implement this.
    
    # Map block index j (0..3) to (row_block, col_block)
    # j=0: (0,0), j=1: (0,1), j=2: (1,0), j=3: (1,1)
    # Map element index i (0..3) to (row_local, col_local)
    # i=0: (0,0), i=1: (0,1), i=2: (1,0), i=3: (1,1)
    
    for k_outer in range(7):
        for k_inner in range(7):
            # U
            for j in range(4):
                if alpha[k_outer, j] == 0: continue
                for i in range(4):
                    if alpha[k_inner, i] == 0: continue
                    # A_block_j is at global (rb*2, cb*2)
                    rb, cb = j // 2, j % 2
                    # Element i is at local (ri, ci)
                    ri, ci = i // 2, i % 2
                    # Global A index: (rb*2 + ri, cb*2 + ci)
                    u_idx = (rb * 2 + ri) * 4 + (cb * 2 + ci)
                    U[u_idx, col] += alpha[k_outer, j] * alpha[k_inner, i]
            
            # V
            # B1 blocks are at cols 0-3.
            for j in range(4):
                if beta[k_outer, j] == 0: continue
                for i in range(4):
                    if beta[k_inner, i] == 0: continue
                    rb, cb = j // 2, j % 2
                    ri, ci = i // 2, i % 2
                    # Global B1 index: (rb*2 + ri, cb*2 + ci)
                    # B1 is cols 0-3, so col is cb*2 + ci.
                    v_idx = (rb * 2 + ri) * 6 + (cb * 2 + ci)
                    V[v_idx, col] += beta[k_outer, j] * beta[k_inner, i]
            
            # W
            # C1 blocks are at cols 0-3.
            # C_comb = sum_k_inner recon[pos, k_inner] * (A_inner * B_inner)
            # The result of the inner multiplication is a 2x2 matrix C_inner.
            # C_inner_rc = sum_i recon[i, k_inner] * M_i.
            # But M_i is a scalar product of A_inner_i and B_inner_i?
            # No, the inner Strassen computes the 2x2 matrix product A_comb * B_comb.
            # The output C_comb is a 2x2 matrix.
            # C_comb_rc = sum_{k_inner} recon[rc_idx, k_inner] * (Inner_Product_k_inner).
            # Where Inner_Product_k_inner is the result of the k_inner-th term of the 2x2 Strassen.
            # Wait, the 2x2 Strassen computes A_comb * B_comb = C_comb.
            # The 7 terms of the 2x2 Strassen are M_1..M_7.
            # C_comb_rc = sum_k gamma_rc_k M_k.
            # Each M_k is a scalar? No, in the 2x2 Strassen, A_comb and B_comb are 2x2 matrices?
            # NO. The 4x4 Strassen treats A_top and B_top as 2x2 matrices of 2x2 blocks.
            # The "multiplication" of these 2x2 matrices is done via 7 multiplications of 2x2 blocks.
            # So the inner step is: Multiply two 2x2 matrices (the combined blocks) using Strassen.
            # The inputs to the inner Strassen are 2x2 matrices (the combined blocks).
            # The outputs are 2x2 matrices (the partial products).
            # So for the inner term k_inner, we compute M_k_inner = A_comb_inner * B_comb_inner?
            # No.
            # Let's clarify the hierarchy.
            # Level 1: 4x4 = 2x2 of 2x2 blocks.
            # Level 2: 2x2 = 2x2 of scalars.
            
            # In Level 1 Strassen:
            # We form 7 linear combinations of the 2x2 blocks of A. Let's call them A'_1..A'_7. Each A'_k is a 2x2 matrix.
            # We form 7 linear combinations of the 2x2 blocks of B. Let's call them B'_1..B'_7. Each B'_k is a 2x2 matrix.
            # We compute M_k = A'_k * B'_k. This is a 2x2 matrix multiplication.
            # To compute M_k, we use Level 2 Strassen.
            # Level 2 Strassen decomposes the multiplication of two 2x2 matrices (A'_k and B'_k) into 7 scalar multiplications.
            # Let the scalar multiplications be m_{k, l} for l=1..7.
            # Then M_k (2x2 matrix) is reconstructed from m_{k, l}.
            # Finally, C (4x4 matrix) is reconstructed from M_1..M_7.
            
            # So, for a fixed (k_outer, k_inner):
            # k_outer is the Level 1 term index (1..7).
            # k_inner is the Level 2 term index (1..7).
            
            # U part:
            # A'_k_outer = sum_j alpha[k_outer, j] A_block_j.
            # A'_k_outer is a 2x2 matrix.
            # The Level 2 Strassen uses A'_k_outer as its input A.
            # The Level 2 term k_inner uses linear combinations of the elements of A'_k_outer.
            # U_{k_inner} = sum_i alpha[k_inner, i] * (element i of A'_k_outer).
            # Element i of A'_k_outer = sum_j alpha[k_outer, j] * (element i of A_block_j).
            # So U_total = sum_i sum_j alpha[k_inner, i] * alpha[k_outer, j] * (element i of A_block_j).
            # This matches my previous derivation.
            
            # V part:
            # Similarly, V_total = sum_i sum_j beta[k_inner, i] * beta[k_outer, j] * (element i of B_block_j).
            
            # W part:
            # The result of Level 2 term k_inner is a scalar m_{k_outer, k_inner}.
            # This scalar contributes to the 2x2 matrix M_{k_outer}.
            # M_{k_outer} is a 2x2 matrix.
            # The elements of M_{k_outer} are linear combinations of m_{k_outer, 1..7}.
            # Specifically, M_{k_outer} = C_comb for the input A'_k_outer, B'_k_outer.
            # So M_{k_outer, rc} = sum_l recon[rc_idx, l] * m_{k_outer, l}.
            # Here l is k_inner.
            # So M_{k_outer, rc} = recon[rc_idx, k_inner] * m_{k_outer, k_inner}.
            
            # Now, C (4x4) is reconstructed from M_1..M_7.
            # C_block_rc = sum_k_outer recon[rc_idx, k_outer] * M_{k_outer}.
            # So C_block_rc = sum_k_outer recon[rc_idx, k_outer] * (recon[rc_idx_inner, k_inner] * m_{k_outer, k_inner}).
            # Wait, the reconstruction of C from M's uses the same recon matrix.
            # Let's check indices.
            # C_block_00 = M_1 + M_4 - M_5 + M_7.
            # M_k is a 2x2 matrix.
            # C_block_00 (2x2) = M_1 + M_4 - M_5 + M_7.
            # Each element of C_block_00 is the sum of corresponding elements of M_k.
            # C_global_rc = sum_k_outer recon[rc_idx, k_outer] * M_{k_outer, rc}.
            # M_{k_outer, rc} = sum_k_inner recon[rc_idx, k_inner] * m_{k_outer, k_inner}.
            # So C_global_rc = sum_k_outer sum_k_inner recon[rc_idx, k_outer] * recon[rc_idx, k_inner] * m_{k_outer, k_inner}.
            
            # The tensor term is m_{k_outer, k_inner}.
            # The coefficient in W for this term is recon[rc_idx, k_outer] * recon[rc_idx, k_inner].
            # rc_idx is the position in the 2x2 block (0..3).
            # Global C index: (rb*2 + ri, cb*2 + ci).
            
            for rc_idx in range(4):
                if recon[rc_idx, k_outer] == 0: continue
                if recon[rc_idx, k_inner] == 0: continue
                ri, ci = rc_idx // 2, rc_idx % 2
                # The M_{k_outer} matrix is placed in the block corresponding to rc_idx?
                # No. M_k is a 2x2 matrix that contributes to ALL C blocks?
                # No. In Strassen, M_k is a 2x2 matrix of blocks?
                # No. A'_k is a 2x2 matrix of SCALARS?
                # Let's re-read Strassen.
                # A is 2x2 of blocks. A'_1 = A_11 + A_22. A'_1 is a 2x2 block (matrix).
                # B'_1 = B_11 + B_22. B'_1 is a 2x2 block.
                # M_1 = A'_1 * B'_1. M_1 is a 2x2 block.
                # C_11 = M_1 + M_4 - M_5 + M_7.
                # C_11 is a 2x2 block.
                # So M_1 contributes to C_11, C_12, C_21, C_22?
                # No. C_11 is the top-left block.
                # C_11 = M_1 + M_4 - M_5 + M_7.
                # C_12 = M_3 + M_5.
                # C_21 = M_2 + M_4.
                # C_22 = M_1 - M_2 + M_3 + M_6.
                # So M_k contributes to specific C blocks.
                # M_k is a 2x2 matrix.
                # The elements of M_k are added to the corresponding elements of the C blocks.
                # So for a fixed k_outer, M_{k_outer} is a 2x2 matrix.
                # Its element (ri, ci) is added to C_block_{block_row, block_col} at local position (ri, ci).
                # Which C blocks?
                # C_block_{0,0} gets M_1[ri,ci] * 1, M_4[ri,ci] * 1, M_5[ri,ci] * -1, M_7[ri,ci] * 1.
                # C_block_{0,1} gets M_3[ri,ci] * 1, M_5[ri,ci] * 1.
                # C_block_{1,0} gets M_2[ri,ci] * 1, M_4[ri,ci] * 1.
                # C_block_{1,1} gets M_1[ri,ci] * 1, M_2[ri,ci] * -1, M_3[ri,ci] * 1, M_6[ri,ci] * 1.
                
                # So, for a given (k_outer, k_inner, ri, ci):
                # The value m_{k_outer, k_inner} contributes to:
                # C_block_{0,0} at (ri,ci) with weight recon[0, k_outer] * recon[ri_idx, k_inner]?
                # No.
                # M_{k_outer}[ri,ci] = sum_l recon[ri_idx, l] * m_{k_outer, l}.
                # So m_{k_outer, k_inner} contributes to M_{k_outer}[ri,ci] with weight recon[ri_idx, k_inner].
                # Then M_{k_outer}[ri,ci] contributes to C_block_{br, bc}[ri,ci] with weight recon[bc_idx, k_outer]?
                # No. The reconstruction of C blocks from M's is:
                # C_block_{0,0} = M_1 + M_4 - M_5 + M_7.
                # So M_1 contributes to C_block_{0,0} with weight 1.
                # M_4 contributes to C_block_{0,0} with weight 1.
                # etc.
                # So for a specific C block (br, bc), the weight from M_{k_outer} is recon[bc_idx, k_outer], where bc_idx is the index of the C block (0..3).
                # Wait, recon is defined for 2x2 matrix multiplication.
                # recon[c_idx, k] gives the weight of M_k in C_c_idx.
                # c_idx 0..3 corresponds to C_00, C_01, C_10, C_11.
                # So for C_block_{br, bc}, the index is c_idx = br*2 + bc.
                # The weight is recon[c_idx, k_outer].
                
                # So total weight for m_{k_outer, k_inner} in C_block_{br, bc}[ri, ci] is:
                # recon[c_idx, k_outer] * recon[ri_idx, k_inner].
                # Where c_idx = br*2 + bc.
                # And ri_idx = ri*2 + ci? No.
                # ri, ci are local indices in the 2x2 block M_{k_outer}.
                # The index in recon for the inner multiplication is ri_idx = ri*2 + ci.
                
                # So we need to sum over all C blocks (br, bc) that have non-zero recon[c_idx, k_outer].
                
                for c_idx in range(4):
                    if recon[c_idx, k_outer] == 0: continue
                    if recon[rc_idx, k_inner] == 0: continue
                    br, bc = c_idx // 2, c_idx % 2
                    # Global C index: (br*2 + ri, bc*2 + ci)
                    w_idx = (br * 2 + ri) * 6 + (bc * 2 + ci)
                    W[w_idx, col] += recon[c_idx, k_outer] * recon[rc_idx, k_inner]
            
            col += 1
    
    # Part 2: A * B2 (4x4 * 4x2)
    # B2 is cols 4-5.
    # C2 is cols 4-5.
    # C2 = A * B2.
    # A = [[A00, A01], [A10, A11]].
    # B2 = [[B2_0], [B2_1]] where B2_0, B2_1 are 2x2.
    # C2 = [[C2_0], [C2_1]] where C2_0, C2_1 are 2x2.
    # C2_0 = A00 B2_0 + A01 B2_1.
    # C2_1 = A10 B2_0 + A11 B2_1.
    
    # We compute 4 products:
    # 1. A00 * B2_0 -> adds to C2_0
    # 2. A01 * B2_1 -> adds to C2_0
    # 3. A10 * B2_0 -> adds to C2_1
    # 4. A11 * B2_1 -> adds to C2_1
    
    # Each product is 2x2 * 2x2 -> 2x2.
    # We use Strassen (rank 7) for each.
    
    terms = [
        # (a_blk, b_blk_row, b_blk_col, c_blk_row, c_blk_col)
        # a_blk: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # b_blk: B2_0 is (0, 4-5), B2_1 is (1, 4-5).
        # c_blk: C2_0 is (0, 4-5), C2_1 is (1, 4-5).
        (0, 0, 0), # A00 * B2_0 -> C2_0
        (1, 1, 0), # A01 * B2_1 -> C2_0
        (2, 0, 1), # A10 * B2_0 -> C2_1
        (3, 1, 1)  # A11 * B2_1 -> C2_1
    ]
    
    for t in terms:
        a_blk, b_blk, c_blk = t
        # a_blk: 0..3
        # b_blk: 0 (B2_0) or 1 (B2_1)
        # c_blk: 0 (C2_0) or 1 (C2_1)
        
        # A block global row/col
        ar, ac = a_blk // 2, a_blk % 2
        # B block global row/col
        # B2_0: rows 0-1, cols 4-5.
        # B2_1: rows 2-3, cols 4-5.
        br = b_blk * 2
        bc = 4
        # C block global row/col
        # C2_0: rows 0-1, cols 4-5.
        # C2_1: rows 2-3, cols 4-5.
        cr = c_blk * 2
        cc = 4
        
        # Use Strassen for this 2x2 * 2x2 product.
        for k in range(7):
            # U
            for i in range(4):
                if alpha[k, i] != 0:
                    ri, ci = i // 2, i % 2
                    u_idx = (ar * 2 + ri) * 4 + (ac * 2 + ci)
                    U[u_idx, col] += alpha[k, i]
            
            # V
            for i in range(4):
                if beta[k, i] != 0:
                    ri, ci = i // 2, i % 2
                    v_idx = (br + ri) * 6 + (bc + ci)
                    V[v_idx, col] += beta[k, i]
            
            # W
            for i in range(4):
                if recon[i, k] != 0:
                    ri, ci = i // 2, i % 2
                    w_idx = (cr + ri) * 6 + (cc + ci)
                    W[w_idx, col] += recon[i, k]
            
            col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
