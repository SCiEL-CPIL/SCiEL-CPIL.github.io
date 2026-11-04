def construct():
    n, m, p = 4, 4, 9
    
    # Total rank: 49 (for first 4 cols) + 49 (for next 4 cols) + 16 (for last col) = 114
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_off, b_off, c_off are offsets in the 4x4 matrices for the 2x2 blocks
    # The 2x2 blocks are indexed by their top-left corner in the 4x4 matrix
    # But here we are doing 4x4x4, so we partition into 2x2 blocks.
    
    # Let's create a function to add a Strassen 2x2x2 product
    # where the inputs are linear combinations of the 2x2 blocks of A and B
    # and the output is added to the 2x2 blocks of C
    
    # Actually, it's easier to just hardcode the 49 terms or generate them.
    
    # Let's define the Strassen 2x2x2 base case (rank 7)
    # U_base, V_base, W_base are 4x7
    
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(U, V, W, a_off, b_off, c_off, col_start):
        c = col_start
        # M1
        U[a_off+0][c] += 1; U[a_off+3][c] += 1
        V[b_off+0][c] += 1; V[b_off+3][c] += 1
        W[c_off+0][c] += 1; W[c_off+3][c] += 1
        c += 1
        # M2
        U[a_off+2][c] += 1; U[a_off+3][c] += 1
        V[b_off+0][c] += 1
        W[c_off+2][c] += 1; W[c_off+3][c] -= 1
        c += 1
        # M3
        U[a_off+0][c] += 1
        V[b_off+2][c] += 1; V[b_off+3][c] -= 1
        W[c_off+1][c] += 1; W[c_off+3][c] += 1
        c += 1
        # M4
        U[a_off+3][c] += 1
        V[b_off+1][c] += 1; V[b_off+0][c] -= 1
        W[c_off+0][c] += 1; W[c_off+2][c] += 1
        c += 1
        # M5
        U[a_off+0][c] += 1; U[a_off+1][c] += 1
        V[b_off+3][c] += 1
        W[c_off+0][c] -= 1; W[c_off+1][c] += 1
        c += 1
        # M6
        U[a_off+2][c] += 1; U[a_off+0][c] -= 1
        V[b_off+0][c] += 1; V[b_off+1][c] += 1
        W[c_off+3][c] += 1
        c += 1
        # M7
        U[a_off+1][c] += 1; U[a_off+3][c] -= 1
        V[b_off+2][c] += 1; V[b_off+3][c] += 1
        W[c_off+0][c] += 1
        c += 1
        return c

    # Now for 4x4x4, we use Strassen on 2x2 blocks.
    # A is 4x4, partitioned into 4 2x2 blocks.
    # A11: rows 0-1, cols 0-1. Offset 0.
    # A12: rows 0-1, cols 2-3. Offset 2.
    # A21: rows 2-3, cols 0-1. Offset 8.
    # A22: rows 2-3, cols 2-3. Offset 10.
    # Note: U index for A[i,j] is i*4 + j.
    # A11 elements: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
    # So for a 2x2 block starting at row r, col c of the 4x4 matrix:
    # The elements are at U[r*4+c], U[r*4+c+1], U[(r+1)*4+c], U[(r+1)*4+c+1].
    # My add_strassen_2x2 assumes the 4 elements are contiguous?
    # No, in add_strassen_2x2, I used a_off+0, a_off+1, a_off+2, a_off+3.
    # This implies the 2x2 matrix is stored contiguously in memory.
    # But in U, A[i,j] is at i*4+j.
    # So A11 is at 0, 1, 4, 5. NOT contiguous.
    # I need to map the logical 2x2 block indices to the physical U indices.
    
    # Let's define a helper to get the physical indices for a 2x2 block.
    def get_block_indices(base_offset, row, col, m=4):
        # base_offset is the offset of the 4x4 matrix in U/V/W
        # row, col are the top-left corner of the 2x2 block within the 4x4
        # Returns list of 4 indices: [top-left, top-right, bottom-left, bottom-right]
        r0 = base_offset + row * m + col
        r1 = base_offset + (row + 1) * m + col
        return [r0, r0 + 1, r1, r1 + 1]
    
    # Modify add_strassen_2x2 to take a list of indices?
    # Or just pass the offsets and calculate inside?
    # It's cleaner to pass the 4 indices for U, 4 for V, 4 for W.
    
    def add_strassen_2x2_explicit(U, V, W, u_idx, v_idx, w_idx, col_start):
        c = col_start
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # M1 = (A00+A11)(B00+B11)
        U[u_idx[0]][c] += 1; U[u_idx[3]][c] += 1
        V[v_idx[0]][c] += 1; V[v_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        c += 1
        
        # M2 = (A10+A11)B00
        U[u_idx[2]][c] += 1; U[u_idx[3]][c] += 1
        V[v_idx[0]][c] += 1
        W[w_idx[2]][c] += 1; W[w_idx[3]][c] -= 1
        c += 1
        
        # M3 = A00(B10-B11)
        U[u_idx[0]][c] += 1
        V[v_idx[2]][c] += 1; V[v_idx[3]][c] -= 1
        W[w_idx[1]][c] += 1; W[w_idx[3]][c] += 1
        c += 1
        
        # M4 = A11(B01-B00)
        U[u_idx[3]][c] += 1
        V[v_idx[1]][c] += 1; V[v_idx[0]][c] -= 1
        W[w_idx[0]][c] += 1; W[w_idx[2]][c] += 1
        c += 1
        
        # M5 = (A00+A01)B11
        U[u_idx[0]][c] += 1; U[u_idx[1]][c] += 1
        V[v_idx[3]][c] += 1
        W[w_idx[0]][c] -= 1; W[w_idx[1]][c] += 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_idx[2]][c] += 1; U[u_idx[0]][c] -= 1
        V[v_idx[0]][c] += 1; V[v_idx[1]][c] += 1
        W[w_idx[3]][c] += 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_idx[1]][c] += 1; U[u_idx[3]][c] -= 1
        V[v_idx[2]][c] += 1; V[v_idx[3]][c] += 1
        W[w_idx[0]][c] += 1
        c += 1
        
        return c

    # Now implement 4x4x4 using Strassen on blocks.
    # A blocks:
    # A11: row 0, col 0. Indices: get_block_indices(0, 0, 0)
    # A12: row 0, col 2. Indices: get_block_indices(0, 0, 2)
    # A21: row 2, col 0. Indices: get_block_indices(0, 2, 0)
    # A22: row 2, col 2. Indices: get_block_indices(0, 2, 2)
    
    # B blocks (for the 4x4 part of B, cols 0-3):
    # B11: row 0, col 0. Indices: get_block_indices(0, 0, 0)
    # ...
    
    # C blocks:
    # C11: row 0, col 0. Indices: get_block_indices(0, 0, 0)
    # ...
    
    def add_strassen_4x4x4(U, V, W, a_off, b_off, c_off, col_start):
        c = col_start
        
        # Get indices for A blocks
        u11 = get_block_indices(a_off, 0, 0)
        u12 = get_block_indices(a_off, 0, 2)
        u21 = get_block_indices(a_off, 2, 0)
        u22 = get_block_indices(a_off, 2, 2)
        
        # Get indices for B blocks
        v11 = get_block_indices(b_off, 0, 0)
        v12 = get_block_indices(b_off, 0, 2)
        v21 = get_block_indices(b_off, 2, 0)
        v22 = get_block_indices(b_off, 2, 2)
        
        # Get indices for C blocks
        w11 = get_block_indices(c_off, 0, 0)
        w12 = get_block_indices(c_off, 0, 2)
        w21 = get_block_indices(c_off, 2, 0)
        w22 = get_block_indices(c_off, 2, 2)
        
        # Strassen products for 2x2 block matrix multiplication
        # M1 = (A11+A22)(B11+B22)
        # We need to form (A11+A22) and (B11+B22) and multiply them using Strassen 2x2x2.
        # But add_strassen_2x2_explicit expects single 2x2 matrices.
        # We can't just sum the indices. We need to create linear combinations.
        # This means we need to add the contributions to U and V for the sum.
        
        # Instead of calling add_strassen_2x2_explicit which hardcodes +/- 1,
        # I should create a generic function that takes coefficients.
        
        # Let's rewrite add_strassen_2x2 to accept coefficient lists.
        # Or, simpler: just add the terms manually for each of the 7 Strassen steps.
        # Each step is a 2x2x2 multiplication.
        # M1 = (A11+A22) * (B11+B22)
        # Let X = A11+A22, Y = B11+B22.
        # X is a 2x2 matrix. Y is a 2x2 matrix.
        # We compute Z = X*Y using Strassen 2x2x2.
        # The U entries for X will be the sum of U entries for A11 and A22.
        # The V entries for Y will be the sum of V entries for B11 and B22.
        # The W entries for Z will be added to C11.
        
        # I will write a helper `add_strassen_2x2_linear` that takes:
        # u_coeffs: list of 4 lists of (index, coeff) for the 4 elements of X
        # v_coeffs: list of 4 lists of (index, coeff) for the 4 elements of Y
        # w_indices: list of 4 indices for the 4 elements of Z (to add to)
        
        def add_strassen_2x2_linear(U, V, W, u_coeffs, v_coeffs, w_indices, col_start):
            c = col_start
            # u_coeffs[i] is a list of (u_idx, coeff) for element i of X
            # v_coeffs[i] is a list of (v_idx, coeff) for element i of Y
            
            # M1 = (X00+X11)(Y00+Y11)
            for idx, coeff in u_coeffs[0]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[3]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] += coeff
            W[w_indices[0]][c] += 1; W[w_indices[3]][c] += 1
            c += 1
            
            # M2 = (X10+X11)Y00
            for idx, coeff in u_coeffs[2]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[3]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] += coeff
            W[w_indices[2]][c] += 1; W[w_indices[3]][c] -= 1
            c += 1
            
            # M3 = X00(Y10-Y11)
            for idx, coeff in u_coeffs[0]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[2]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] -= coeff
            W[w_indices[1]][c] += 1; W[w_indices[3]][c] += 1
            c += 1
            
            # M4 = X11(Y01-Y00)
            for idx, coeff in u_coeffs[3]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[1]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] -= coeff
            W[w_indices[0]][c] += 1; W[w_indices[2]][c] += 1
            c += 1
            
            # M5 = (X00+X01)Y11
            for idx, coeff in u_coeffs[0]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[1]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] += coeff
            W[w_indices[0]][c] -= 1; W[w_indices[1]][c] += 1
            c += 1
            
            # M6 = (X10-X00)(Y00+Y01)
            for idx, coeff in u_coeffs[2]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[0]: U[idx][c] -= coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[1]: V[idx][c] += coeff
            W[w_indices[3]][c] += 1
            c += 1
            
            # M7 = (X01-X11)(Y10+Y11)
            for idx, coeff in u_coeffs[1]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[3]: U[idx][c] -= coeff
            for idx, coeff in v_coeffs[2]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] += coeff
            W[w_indices[0]][c] += 1
            c += 1
            
            return c
        
        # Helper to create coeffs for a single block
        def block_coeffs(indices):
            return [[(idx, 1)] for idx in indices]
            
        # Helper to create coeffs for sum of two blocks
        def sum_block_coeffs(idx1, idx2):
            return [
                [(idx1[0], 1), (idx2[0], 1)],
                [(idx1[1], 1), (idx2[1], 1)],
                [(idx1[2], 1), (idx2[2], 1)],
                [(idx1[3], 1), (idx2[3], 1)]
            ]
            
        # Helper to create coeffs for diff of two blocks
        def diff_block_coeffs(idx1, idx2):
            return [
                [(idx1[0], 1), (idx2[0], -1)],
                [(idx1[1], 1), (idx2[1], -1)],
                [(idx1[2], 1), (idx2[2], -1)],
                [(idx1[3], 1), (idx2[3], -1)]
            ]

        # M1 = (A11+A22)(B11+B22) -> C11
        u_x = sum_block_coeffs(u11, u22)
        v_y = sum_block_coeffs(v11, v22)
        c = add_strassen_2x2_linear(U, V, W, u_x, v_y, w11, c)
        
        # M2 = (A21+A22)B11 -> C21, C22
        u_x = sum_block_coeffs(u21, u22)
        v_y = block_coeffs(v11)
        # M2 contributes to C21 (as +M2) and C22 (as -M2)
        # Wait, Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M2 adds +1 to C21 and -1 to C22.
        # My add_strassen_2x2_linear adds to w_indices.
        # I need to call it twice? Or modify it to take multiple w targets with coeffs.
        # Let's just call it for C21 with +1 coeffs and C22 with -1 coeffs?
        # No, the scalar multiplication is the same. The W matrix entries just accumulate.
        # So I can call add_strassen_2x2_linear with w_indices = w21, and then again with w_indices = w22?
        # No, that would double the rank.
        # The rank is determined by the number of scalar multiplications.
        # Each call to add_strassen_2x2_linear uses 7 scalar multiplications.
        # I need to add the results to C21 and C22 simultaneously.
        # I should modify add_strassen_2x2_linear to accept a list of (w_indices, coeff) pairs.
        
        # Let's refactor add_strassen_2x2_linear to accept w_targets: list of (indices, coeff)
        
        # Re-defining inside the function for clarity in the patch
        # (I will just implement the logic directly in the patch)
        
        # I will use the version that takes w_targets.
        
        return c
    
    # This is getting too complex for a single edit block without testing.
    # I will use a simpler approach: just use Schoolbook for the 4x4x4 part?
    # Rank 64 for 4x4x4.
    # 64 + 64 + 16 = 144. No improvement.
    
    # I will use Strassen for 4x4x4 (Rank 49).
    # I will implement the 49-term decomposition explicitly by calling the 2x2 Strassen 7 times,
    # but I must handle the W accumulation correctly.
    
    # Let's assume I can write a robust `add_strassen_4x4x4` that does this.
    
    # For the 4x4x5 part, I will use Schoolbook (Rank 80).
    # Total 49 + 80 = 129.
    
    # Let's implement the 4x4x4 Strassen part carefully.
    
    def get_block_indices(base_offset, row, col, m=4):
        r0 = base_offset + row * m + col
        r1 = base_offset + (row + 1) * m + col
        return [r0, r0 + 1, r1, r1 + 1]
        
    def add_strassen_4x4x4(U, V, W, a_off, b_off, c_off, col_start):
        c = col_start
        u11 = get_block_indices(a_off, 0, 0)
        u12 = get_block_indices(a_off, 0, 2)
        u21 = get_block_indices(a_off, 2, 0)
        u22 = get_block_indices(a_off, 2, 2)
        
        v11 = get_block_indices(b_off, 0, 0)
        v12 = get_block_indices(b_off, 0, 2)
        v21 = get_block_indices(b_off, 2, 0)
        v22 = get_block_indices(b_off, 2, 2)
        
        w11 = get_block_indices(c_off, 0, 0)
        w12 = get_block_indices(c_off, 0, 2)
        w21 = get_block_indices(c_off, 2, 0)
        w22 = get_block_indices(c_off, 2, 2)
        
        # Helper to add 2x2 Strassen with linear combinations
        def add_2x2_strassen(u_coeffs, v_coeffs, w_targets, col_start):
            # w_targets: list of (w_indices, coeff)
            c = col_start
            # M1
            for idx, coeff in u_coeffs[0]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[3]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] += coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[0]][c] += w_coeff
                W[w_idx[3]][c] += w_coeff
            c += 1
            
            # M2
            for idx, coeff in u_coeffs[2]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[3]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] += coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[2]][c] += w_coeff
                W[w_idx[3]][c] -= w_coeff
            c += 1
            
            # M3
            for idx, coeff in u_coeffs[0]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[2]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] -= coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[1]][c] += w_coeff
                W[w_idx[3]][c] += w_coeff
            c += 1
            
            # M4
            for idx, coeff in u_coeffs[3]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[1]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] -= coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[0]][c] += w_coeff
                W[w_idx[2]][c] += w_coeff
            c += 1
            
            # M5
            for idx, coeff in u_coeffs[0]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[1]: U[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] += coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[0]][c] -= w_coeff
                W[w_idx[1]][c] += w_coeff
            c += 1
            
            # M6
            for idx, coeff in u_coeffs[2]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[0]: U[idx][c] -= coeff
            for idx, coeff in v_coeffs[0]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[1]: V[idx][c] += coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[3]][c] += w_coeff
            c += 1
            
            # M7
            for idx, coeff in u_coeffs[1]: U[idx][c] += coeff
            for idx, coeff in u_coeffs[3]: U[idx][c] -= coeff
            for idx, coeff in v_coeffs[2]: V[idx][c] += coeff
            for idx, coeff in v_coeffs[3]: V[idx][c] += coeff
            for w_idx, w_coeff in w_targets:
                W[w_idx[0]][c] += w_coeff
            c += 1
            return c
            
        def block_coeffs(indices):
            return [[(idx, 1)] for idx in indices]
            
        def sum_block_coeffs(idx1, idx2):
            return [
                [(idx1[0], 1), (idx2[0], 1)],
                [(idx1[1], 1), (idx2[1], 1)],
                [(idx1[2], 1), (idx2[2], 1)],
                [(idx1[3], 1), (idx2[3], 1)]
            ]
            
        def diff_block_coeffs(idx1, idx2):
            return [
                [(idx1[0], 1), (idx2[0], -1)],
                [(idx1[1], 1), (idx2[1], -1)],
                [(idx1[2], 1), (idx2[2], -1)],
                [(idx1[3], 1), (idx2[3], -1)]
            ]

        # M1 = (A11+A22)(B11+B22)
        # C11 += M1
        u_x = sum_block_coeffs(u11, u22)
        v_y = sum_block_coeffs(v11, v22)
        c = add_2x2_strassen(u_x, v_y, [(w11, 1)], c)
        
        # M2 = (A21+A22)B11
        # C21 += M2, C22 -= M2
        u_x = sum_block_coeffs(u21, u22)
        v_y = block_coeffs(v11)
        c = add_2x2_strassen(u_x, v_y, [(w21, 1), (w22, -1)], c)
        
        # M3 = A11(B21-B22)
        # C12 += M3, C22 += M3
        u_x = block_coeffs(u11)
        v_y = diff_block_coeffs(v21, v22)
        c = add_2x2_strassen(u_x, v_y, [(w12, 1), (w22, 1)], c)
        
        # M4 = A22(B12-B11)
        # C11 += M4, C21 += M4
        u_x = block_coeffs(u22)
        v_y = diff_block_coeffs(v12, v11)
        c = add_2x2_strassen(u_x, v_y, [(w11, 1), (w21, 1)], c)
        
        # M5 = (A11+A12)B22
        # C11 -= M5, C12 += M5
        u_x = sum_block_coeffs(u11, u12)
        v_y = block_coeffs(v22)
        c = add_2x2_strassen(u_x, v_y, [(w11, -1), (w12, 1)], c)
        
        # M6 = (A21-A11)(B11+B12)
        # C22 += M6
        u_x = diff_block_coeffs(u21, u11)
        v_y = sum_block_coeffs(v11, v12)
        c = add_2x2_strassen(u_x, v_y, [(w22, 1)], c)
        
        # M7 = (A12-A22)(B21+B22)
        # C11 += M7
        u_x = diff_block_coeffs(u12, u22)
        v_y = sum_block_coeffs(v21, v22)
        c = add_2x2_strassen(u_x, v_y, [(w11, 1)], c)
        
        return c

    # Add first 4x4x4 block (cols 0-3 of B)
    col = add_strassen_4x4x4(U, V, W, 0, 0, 0, col)
    
    # Add second 4x4x4 block (cols 4-7 of B)
    col = add_strassen_4x4x4(U, V, W, 0, 4, 4, col)
    
    # Add last 4x4x1 block (col 8 of B) using Schoolbook
    # A is 4x4, B is 4x1. C is 4x1.
    # C[i, 0] = sum_j A[i, j] * B[j, 0]
    # Rank 16.
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 4 + 8][col] = 1  # B[j, 8] is at V[j*9+8]
            W[8 * 4 + i][col] = 1  # C[8, i] is at W[8*4+i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
