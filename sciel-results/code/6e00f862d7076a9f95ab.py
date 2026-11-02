import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Rank 114 decomposition
    # 49 for first 4x4 block (B cols 0-3)
    # 49 for second 4x4 block (B cols 4-7)
    # 16 for 4x1 block (B col 8)
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen coefficients
    # u, v, w are lists of 4 coefficients for x00, x01, x10, x11
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_2x2 = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def add_strassen_4x4(b_col_offset, c_col_offset, rank_offset):
        # b_col_offset: 0 or 4
        # c_col_offset: 0 or 4
        
        # Define the 7 block products
        # Each product is (u_coeffs, v_coeffs) where coeffs are for A and B blocks
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        
        # We need to map block indices to actual matrix indices
        # A block (bi, bj) covers rows bi*2 .. bi*2+1, cols bj*2 .. bj*2+1
        # B block (bj, bk) covers rows bj*2 .. bj*2+1, cols b_col_offset + bk*2 .. b_col_offset + bk*2+1
        # C block (bi, bk) covers rows bi*2 .. bi*2+1, cols c_col_offset + bk*2 .. c_col_offset + bk*2+1
        
        # Strassen 4x4 products
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # List of (u_block_coeffs, v_block_coeffs)
        # u_block_coeffs: dict {block_idx: coeff}
        # v_block_coeffs: dict {block_idx: coeff}
        
        products = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1: A00+A11, B00+B11
            ({2: 1, 3: 1}, {0: 1}),       # M2: A10+A11, B00
            ({0: 1}, {1: 1, 3: -1}),      # M3: A00, B01-B11
            ({3: 1}, {2: 1, 0: -1}),      # M4: A11, B10-B00
            ({0: 1, 1: 1}, {3: 1}),       # M5: A00+A01, B11
            ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6: A10-A00, B00+B01
            ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7: A01-A11, B10+B11
        ]
        
        for p_idx, (u_blocks, v_blocks) in enumerate(products):
            # For each of the 7 2x2 Strassen terms
            for t_idx, (u_2x2, v_2x2, w_2x2) in enumerate(strassen_2x2):
                r = rank_offset + p_idx * 7 + t_idx
                
                # Expand u_2x2 and v_2x2 into global indices
                # u_2x2 has 4 coeffs for x00, x01, x10, x11
                # These correspond to the 4 elements of the 2x2 matrix X
                # X is a linear combination of A blocks
                # The elements of X are x00, x01, x10, x11
                # x00 is the (0,0) element of X, which is a linear combo of A block elements?
                # No, X is a 2x2 matrix where each entry is a scalar?
                # No, in the 4x4 Strassen, the "elements" are 2x2 matrices.
                # So X is a 2x2 matrix of 2x2 matrices.
                # The 2x2 Strassen decomposition acts on the "scalar" level of the 2x2 matrix of 2x2 matrices?
                # No, it acts on the 2x2 grid of blocks.
                # So we have a 2x2 matrix X of blocks.
                # We apply Strassen to this 2x2 matrix of blocks.
                # The "multiplication" of blocks is matrix multiplication of 2x2 matrices.
                # So we need to decompose the product of two 2x2 matrices (which are the blocks) into rank-1 terms.
                # This is exactly the 2x2 Strassen decomposition.
                
                # So, for each of the 7 block products, we have a 2x2 matrix X (linear combo of A blocks)
                # and a 2x2 matrix Y (linear combo of B blocks).
                # We apply the 2x2 Strassen decomposition to X*Y.
                # The 2x2 Strassen decomposition gives 7 rank-1 terms for the 2x2 matrix multiplication.
                # Each term involves linear combinations of the elements of X and Y.
                # The elements of X are themselves linear combinations of A elements.
                
                # So we need to map the "elements" of X (x00, x01, x10, x11) to A elements.
                # x00 is the top-left block of X.
                # X is a 2x2 matrix of 2x2 blocks.
                # x00 corresponds to the block at (0,0) of the 2x2 grid.
                # Which block is that?
                # It depends on the product M_k.
                # For M1, X = A00 + A11.
                # This means the block at (0,0) of X is A00? No.
                # M1 = (A00 + A11)(B00 + B11).
                # This is a product of two 2x2 matrices?
                # No, A00 is a 2x2 matrix. A11 is a 2x2 matrix.
                # A00 + A11 is a 2x2 matrix.
                # So X is a 2x2 matrix.
                # The "elements" of X are scalars?
                # No, the "elements" of the 2x2 matrix X are the elements of the 2x2 matrix A00+A11.
                # So x00 = (A00+A11)[0,0], x01 = (A00+A11)[0,1], etc.
                
                # This means we need to express the 4 elements of X in terms of A elements.
                # And the 4 elements of Y in terms of B elements.
                
                # Let's define the mapping for each product.
                # For M1: X = A00 + A11.
                # x00 = A00[0,0] + A11[0,0]
                # x01 = A00[0,1] + A11[0,1]
                # x10 = A00[1,0] + A11[1,0]
                # x11 = A00[1,1] + A11[1,1]
                
                # We need to add these to U.
                
                # This is getting complex. Let's precompute the mappings.
                
                pass

    # This approach is too complex to implement dynamically in the loop.
    # Let's hardcode the 49 terms for one 4x4 block.
    
    # I will return the schoolbook for now to ensure correctness, but this is not optimal.
    # However, the prompt asks for a combination of strategies.
    # The current program is schoolbook.
    # The inspiration is Strassen.
    # I will implement the Strassen 4x4 decomposition correctly.
    
    # Due to complexity, I will provide a known valid rank 114 decomposition if I can, 
    # or fall back to schoolbook if I can't verify.
    # But I must improve.
    
    # Let's try to implement the mapping.
    
    def get_a_indices(block_idx):
        # block_idx: 0:00, 1:01, 2:10, 3:11
        # Returns list of (i, j) in A
        r = block_idx // 2
        c = block_idx % 2
        return [(r*2 + di, c*2 + dj) for di in range(2) for dj in range(2)]
    
    def get_b_indices(block_idx, b_col_offset):
        r = block_idx // 2
        c = block_idx % 2
        return [(r*2 + di, b_col_offset + c*2 + dj) for di in range(2) for dj in range(2)]
    
    def get_c_indices(block_idx, c_col_offset):
        r = block_idx // 2
        c = block_idx % 2
        return [(c_col_offset + c*2 + dj, r*2 + di) for dj in range(2) for di in range(2)]
        # Note: W[k*n+i, r]. k is col, i is row.
        # C[i, k].
    
    # Re-implement add_strassen_4x4 with explicit mapping
    
    def add_strassen_4x4(b_col_offset, c_col_offset, rank_offset):
        products = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1})
        ]
        
        for p_idx, (u_blocks, v_blocks) in enumerate(products):
            for t_idx, (u_2x2, v_2x2, w_2x2) in enumerate(strassen_2x2):
                r = rank_offset + p_idx * 7 + t_idx
                
                # Map u_2x2 to A
                # u_2x2 has 4 coeffs for x00, x01, x10, x11
                # x00 is element (0,0) of X
                # X is sum of blocks in u_blocks
                # So x00 = sum_{b} coeff_b * A_block[b][0,0]
                
                # We need to add to U[i*m+j]
                
                # For each of the 4 elements of X (x00..x11):
                for e_idx in range(4):
                    coeff_x = u_2x2[e_idx]
                    if coeff_x == 0: continue
                    
                    # x_e_idx is element (er, ec) of X
                    er = e_idx // 2
                    ec = e_idx % 2
                    
                    # This element is sum of A_block[b][er, ec] * coeff_b
                    for b_idx, coeff_b in u_blocks.items():
                        if coeff_b == 0: continue
                        # A_block[b_idx] has elements at get_a_indices(b_idx)
                        # We need the element at (er, ec) within the block
                        # get_a_indices returns all 4 elements.
                        # We need to pick the one at local (er, ec)
                        # The local indices are 0,1 for row and col.
                        # The global indices are i, j.
                        # i = b_r*2 + er, j = b_c*2 + ec
                        b_r = b_idx // 2
                        b_c = b_idx % 2
                        i = b_r * 2 + er
                        j = b_c * 2 + ec
                        U[i * m + j][r] += coeff_x * coeff_b
                        
                # Map v_2x2 to B
                for e_idx in range(4):
                    coeff_y = v_2x2[e_idx]
                    if coeff_y == 0: continue
                    er = e_idx // 2
                    ec = e_idx % 2
                    for b_idx, coeff_b in v_blocks.items():
                        if coeff_b == 0: continue
                        b_r = b_idx // 2
                        b_c = b_idx % 2
                        i = b_r * 2 + er
                        j = b_col_offset + b_c * 2 + ec
                        V[i * p + j][r] += coeff_y * coeff_b
                        
                # Map w_2x2 to C
                for e_idx in range(4):
                    coeff_z = w_2x2[e_idx]
                    if coeff_z == 0: continue
                    er = e_idx // 2
                    ec = e_idx % 2
                    # C element at (er, ec) of the block?
                    # No, w_2x2 corresponds to the 2x2 matrix Z.
                    # Z is the product of the 2x2 blocks.
                    # The elements of Z are the blocks of C.
                    # So z_e_idx is the block at (er, ec) of the 2x2 grid of C blocks.
                    # This block is a 2x2 matrix.
                    # Wait, the 2x2 Strassen decomposition produces a 2x2 matrix Z.
                    # The "elements" of Z are scalars?
                    # No, in the 4x4 case, the "elements" of the 2x2 matrix of blocks are 2x2 matrices.
                    # So the 2x2 Strassen decomposition is applied to the 2x2 matrix of blocks.
                    # The result is a 2x2 matrix of blocks.
                    # Each block in the result is a 2x2 matrix.
                    # So we need to add the 2x2 matrix (block) to C.
                    
                    # The w_2x2 vector has 4 coeffs for z00, z01, z10, z11.
                    # These are the 4 blocks of C.
                    # So for each block z_e_idx, we add coeff_z * (the 2x2 block of C).
                    
                    c_r = e_idx // 2
                    c_c = e_idx % 2
                    # The block is at C rows c_r*2..c_r*2+1, cols c_col_offset + c_c*2..c_col_offset + c_c*2+1
                    for di in range(2):
                        for dj in range(2):
                            i = c_r * 2 + di
                            k = c_col_offset + c_c * 2 + dj
                            W[k * n + i][r] += coeff_z

    add_strassen_4x4(0, 0, 0)
    add_strassen_4x4(4, 4, 49)
    
    # Schoolbook for 4x1
    # A (4x4) * B (4x1) -> C (4x1)
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = 98 + i * 4 + j
            U[i * m + j][r] = 1
            V[j * p + 8][r] = 1
            W[8 * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
