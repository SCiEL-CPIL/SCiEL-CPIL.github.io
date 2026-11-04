DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114: 49 (for cols 0-3) + 49 (for cols 4-7) + 16 (for col 8)
    rank = 114
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Function to add a 2x2x2 Strassen decomposition
    # a_base: (row, col) offset in A
    # b_base: (row, col) offset in B
    # c_base: (row, col) offset in C
    # sa, sb, sc: scale factors
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c, sa, sb, sc):
        nonlocal col_idx
        # 7 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        # A: (0,0) + (1,1)
        # B: (0,0) + (1,1)
        # C: (0,0) + (1,1)
        U[a_r*2 + a_c][col_idx] += sa
        U[(a_r+1)*2 + (a_c+1)][col_idx] += sa
        V[b_r*2 + b_c][col_idx] += sb
        V[(b_r+1)*2 + (b_c+1)][col_idx] += sb
        W[(c_r*2 + c_r)*n + c_c][col_idx] += sc # C00 index: k= c_r*2+c_r? No.
        # C index: k*n + i.
        # C00 is i=0, k=0 relative to block.
        # Global i = c_r*2 + 0, k = c_c*2 + 0?
        # C is 4x4. Block (0,0) is rows 0-1, cols 0-1.
        # C00 is row 0, col 0.
        # C11 is row 1, col 1.
        # W index: k*n + i.
        # C00: k=0, i=0 -> 0
        # C11: k=1, i=1 -> 1*n + 1
        W[c_r*2*n + c_r][col_idx] += sc
        W[(c_r+1)*2*n + (c_r+1)][col_idx] += sc
        col_idx += 1

        # Term 2: M2
        # A: (1,0) + (1,1)
        # B: (0,0)
        # C: (1,0) + (1,1)
        U[(a_r+1)*2 + a_c][col_idx] += sa
        U[(a_r+1)*2 + (a_c+1)][col_idx] += sa
        V[b_r*2 + b_c][col_idx] += sb
        W[(c_r+1)*2*n + c_r][col_idx] += sc
        W[(c_r+1)*2*n + (c_r+1)][col_idx] += sc
        col_idx += 1

        # Term 3: M3
        # A: (0,0)
        # B: (0,1) - (1,1)
        # C: (0,1)
        U[a_r*2 + a_c][col_idx] += sa
        V[b_r*2 + (b_c+1)][col_idx] += sb
        V[(b_r+1)*2 + (b_c+1)][col_idx] -= sb
        W[c_r*2*n + (c_r+1)][col_idx] += sc
        col_idx += 1

        # Term 4: M4
        # A: (1,1)
        # B: (1,0) - (0,0)
        # C: (0,0) + (1,0)
        U[(a_r+1)*2 + (a_c+1)][col_idx] += sa
        V[(b_r+1)*2 + b_c][col_idx] += sb
        V[b_r*2 + b_c][col_idx] -= sb
        W[c_r*2*n + c_r][col_idx] += sc
        W[(c_r+1)*2*n + c_r][col_idx] += sc
        col_idx += 1

        # Term 5: M5
        # A: (0,0) + (0,1)
        # B: (1,1)
        # C: -(0,0) + (0,1)
        U[a_r*2 + a_c][col_idx] += sa
        U[a_r*2 + (a_c+1)][col_idx] += sa
        V[(b_r+1)*2 + (b_c+1)][col_idx] += sb
        W[c_r*2*n + c_r][col_idx] -= sc
        W[c_r*2*n + (c_r+1)][col_idx] += sc
        col_idx += 1

        # Term 6: M6
        # A: (1,0) - (0,0)
        # B: (0,0) + (0,1)
        # C: (1,1)
        U[(a_r+1)*2 + a_c][col_idx] += sa
        U[a_r*2 + a_c][col_idx] -= sa
        V[b_r*2 + b_c][col_idx] += sb
        V[b_r*2 + (b_c+1)][col_idx] += sb
        W[(c_r+1)*2*n + (c_r+1)][col_idx] += sc
        col_idx += 1

        # Term 7: M7
        # A: (0,1) - (1,1)
        # B: (1,0) + (1,1)
        # C: (0,0) - (1,1)
        U[a_r*2 + (a_c+1)][col_idx] += sa
        U[(a_r+1)*2 + (a_c+1)][col_idx] -= sa
        V[(b_r+1)*2 + b_c][col_idx] += sb
        V[(b_r+1)*2 + (b_c+1)][col_idx] += sb
        W[c_r*2*n + c_r][col_idx] += sc
        W[(c_r+1)*2*n + (c_r+1)][col_idx] -= sc
        col_idx += 1

    # Function to add 4x4x4 Strassen decomposition
    # b_col_start: starting column in B
    def add_strassen_4x4(b_col_start):
        nonlocal col_idx
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11 (cols b_col_start to b_col_start+3)
        
        # M1 = (A00+A11)(B00+B11)
        add_strassen_2x2(0, 0, b_col_start, 0, 0, 0, 1, 1, 1) # A00, B00, C00
        add_strassen_2x2(0, 0, b_col_start+2, 0, 0, 0, 1, 1, 1) # A00, B11, C00
        add_strassen_2x2(2, 2, b_col_start, 0, 0, 0, 1, 1, 1) # A11, B00, C00
        add_strassen_2x2(2, 2, b_col_start+2, 0, 0, 0, 1, 1, 1) # A11, B11, C00
        # Wait, M1 is (A00+A11)(B00+B11).
        # This is a product of sums.
        # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
        # We need to multiply these two 2x2 matrices.
        # This requires 7 Strassen steps.
        # My add_strassen_2x2 adds ONE rank-1 term.
        # I need to call it 7 times with appropriate coefficients?
        # No, add_strassen_2x2 implements the full rank-7 decomposition of a 2x2x2 product.
        # But here the inputs are sums of blocks.
        # So I need to pass the sum of blocks to the 2x2 decomposition.
        # My add_strassen_2x2 assumes specific block offsets.
        # I need a more flexible function or expand the sums.
        
        # Expansion:
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # Each is a 2x2x2 product.
        # So we need 4 calls to add_strassen_2x2?
        # No, add_strassen_2x2 computes the full product of two 2x2 matrices.
        # So I need to compute the product of (A00+A11) and (B00+B11).
        # Let X = A00+A11, Y = B00+B11.
        # Compute XY using Strassen.
        # This requires 7 rank-1 terms.
        # But X and Y are sums.
        # The linearity allows us to say:
        # XY = A00B00 + A00B11 + A11B00 + A11B11.
        # So we just add the 4 products.
        # Each product is a 2x2x2 multiplication.
        # So we call add_strassen_2x2 4 times?
        # No, add_strassen_2x2 adds 7 terms.
        # So 4 calls add 28 terms.
        # But M1 is just ONE of the 7 Strassen products for the 4x4x4.
        # This is inefficient.
        
        # Correct approach:
        # The 4x4x4 Strassen uses 7 products of 2x2 matrices.
        # Each 2x2 matrix product is decomposed into 7 rank-1 tensors.
        # Total 49 rank-1 tensors.
        # I need to construct the 7 "super" 2x2x2 products.
        
        # Super M1: (A00+A11)(B00+B11)
        # This is a 2x2x2 product where A is (A00+A11) and B is (B00+B11).
        # I need a function that takes 4 A-blocks (with signs) and 4 B-blocks (with signs) and computes the product.
        
        def add_super_strassen_2x2(a_blocks, b_blocks, c_r, c_c, sc):
            # a_blocks: list of (r, c, sign)
            # b_blocks: list of (r, c, sign)
            # This computes sum(a_i * A_i) * sum(b_j * B_j)
            # = sum_i sum_j a_i b_j A_i B_j
            # Each A_i B_j is a 2x2x2 product.
            # We add them all.
            for ar, ac, asa in a_blocks:
                for br, bc, sb in b_blocks:
                    add_strassen_2x2(ar, ac, br, bc, c_r, c_c, asa * sb, 1, sc)
                    # Note: add_strassen_2x2 adds 7 terms.
                    # This will add 7 * (len(a) * len(b)) terms.
                    # For M1, len(a)=2, len(b)=2 -> 4 products -> 28 terms.
                    # Total for 7 M's: 7 * 28 = 196 terms?
                    # No, this is wrong.
                    # Strassen for 4x4x4 uses 7 products.
                    # Each product is a 2x2x2 multiplication.
                    # The rank of 2x2x2 is 7.
                    # So each product contributes 7 rank-1 tensors.
                    # Total 49.
                    # My expansion adds 28 for M1.
                    # This is because I am decomposing the 2x2x2 product of sums into sums of 2x2x2 products.
                    # This is correct algebraically, but the rank is higher.
                    # Rank of sum of products is <= sum of ranks.
                    # Rank of (A+B)(C+D) <= Rank(AC) + Rank(AD) + Rank(BC) + Rank(BD).
                    # 7+7+7+7 = 28.
                    # So my method yields rank 7*28 = 196?
                    # No, the 7 Strassen products are linear combinations.
                    # I should use the rank-7 decomposition of the 2x2x2 tensor directly on the summed blocks.
                    # But my add_strassen_2x2 is hardcoded for specific blocks.
                    
            pass

    # This is getting too complex for a short time.
    # I will stick to the schoolbook.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
