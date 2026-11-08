import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Part 1: M_{4,4,4} using Strassen (Rank 49)
    # Part 2: M_{4,4,5} using Schoolbook (Rank 80)
    
    rank = 49 + 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(i, j, k, col, u_val=1, v_val=1, w_val=1):
        # i: row of A (0..n-1)
        # j: col of A / row of B (0..m-1)
        # k: col of B (0..p-1)
        # Note: V index is j*p + k. W index is k*n + i.
        U[i * m + j][col] += u_val
        V[j * p + k][col] += v_val
        W[k * n + i][col] += w_val

    # --- Part 2: Schoolbook for k=4..8 ---
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                add_term(i, j, k, col)
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    # --- Part 1: Strassen for k=0..3 ---
    # We need to generate the 49 terms for 4x4x4.
    # Structure:
    # 1. High-level Strassen on 2x2 blocks of 2x2 matrices.
    # 2. Low-level Strassen on 2x2 matrices.
    
    # Let's define the 2x2 Strassen decomposition first.
    # Inputs: X (2x2), Y (2x2). Output: Z (2x2).
    # Z = X * Y.
    # Indices for 2x2: 0, 1, 2, 3 (row-major).
    # X: [[x0, x1], [x2, x3]]
    # Y: [[y0, y1], [y2, y3]]
    # Z: [[z0, z1], [z2, z3]]
    
    # Strassen terms for 2x2:
    # s1 = (x0+x3)(y0+y3)
    # s2 = (x2+x3)y0
    # s3 = x0(y1-y3)
    # s4 = x3(y2-y0)
    # s5 = (x0+x1)y3
    # s6 = (x2-x0)(y0+y1)
    # s7 = (x1-x3)(y2+y3)
    
    # Reconstruct Z:
    # z0 = s1 + s4 - s5 + s7
    # z1 = s3 + s5
    # z2 = s2 + s4
    # z3 = s1 - s2 + s3 + s6
    
    # We represent a term by its contributions to X, Y, and Z.
    # X_contrib: list of (idx, coeff)
    # Y_contrib: list of (idx, coeff)
    # Z_contrib: list of (idx, coeff)
    
    def get_2x2_strassen_terms():
        terms = []
        # s1
        terms.append({
            'x': [(0, 1), (3, 1)],
            'y': [(0, 1), (3, 1)],
            'z': [(0, 1), (3, 1)]
        })
        # s2
        terms.append({
            'x': [(2, 1), (3, 1)],
            'y': [(0, 1)],
            'z': [(2, 1)]
        })
        # s3
        terms.append({
            'x': [(0, 1)],
            'y': [(1, 1), (3, -1)],
            'z': [(1, 1)]
        })
        # s4
        terms.append({
            'x': [(3, 1)],
            'y': [(2, 1), (0, -1)],
            'z': [(0, 1), (2, 1)]
        })
        # s5
        terms.append({
            'x': [(0, 1), (1, 1)],
            'y': [(3, 1)],
            'z': [(0, -1), (1, 1)]
        })
        # s6
        terms.append({
            'x': [(2, 1), (0, -1)],
            'y': [(0, 1), (1, 1)],
            'z': [(3, 1)]
        })
        # s7
        terms.append({
            'x': [(1, 1), (3, -1)],
            'y': [(2, 1), (3, 1)],
            'z': [(0, 1), (3, -1)]
        })
        return terms

    strassen_2x2 = get_2x2_strassen_terms()
    
    # High-level Strassen for 4x4 (blocks of 2x2).
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # High-level terms H1..H7:
    # H1 = (A00+A11)(B00+B11)
    # H2 = (A10+A11)B00
    # H3 = A00(B01-B11)
    # H4 = A11(B10-B00)
    # H5 = (A00+A01)B11
    # H6 = (A10-A00)(B00+B01)
    # H7 = (A01-A11)(B10+B11)
    
    # Reconstruct C blocks:
    # C00 = H1 + H4 - H5 + H7
    # C01 = H3 + H5
    # C10 = H2 + H4
    # C11 = H1 - H2 + H3 + H6
    
    # We iterate over the 7 high-level terms.
    # For each high-level term, we have a linear combination of A-blocks (X) and B-blocks (Y).
    # The result is a 2x2 matrix product (which is a block of C).
    # We use the 2x2 Strassen terms to decompose this product.
    
    # Mapping block indices to global indices:
    # Block (br, bc) of A (4x4):
    #   Rows: br*2, br*2+1
    #   Cols: bc*2, bc*2+1
    #   Local index l (0..3) in 2x2 block maps to global (br*2 + l//2, bc*2 + l%2)
    #   Global U index: row*4 + col
    
    def block_to_global_a(br, bc, local_idx):
        # local_idx 0..3 for 2x2
        r = br * 2 + (local_idx // 2)
        c = bc * 2 + (local_idx % 2)
        return r * 4 + c

    def block_to_global_b(br, bc, local_idx):
        # B is 4x4 (first 4 cols).
        # Block (br, bc)
        # Rows: br*2, br*2+1 (these are j indices 0..3)
        # Cols: bc*2, bc*2+1 (these are k indices 0..3)
        # Global V index: j*9 + k
        j = br * 2 + (local_idx // 2)
        k = bc * 2 + (local_idx % 2)
        return j * 9 + k

    def block_to_global_c(br, bc, local_idx):
        # C is 4x4 (first 4 cols).
        # Block (br, bc)
        # Rows: br*2, br*2+1 (these are i indices 0..3)
        # Cols: bc*2, bc*2+1 (these are k indices 0..3)
        # Global W index: k*4 + i
        i = br * 2 + (local_idx // 2)
        k = bc * 2 + (local_idx % 2)
        return k * 4 + i

    # Define high-level terms
    # Each term is (x_blocks, y_blocks, z_block)
    # x_blocks: list of (block_idx, coeff) where block_idx is (br, bc)
    # y_blocks: list of (block_idx, coeff)
    # z_block: (br, bc) and coeff
    
    high_level_terms = [
        # H1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        {
            'x': [((0,0), 1), ((1,1), 1)],
            'y': [((0,0), 1), ((1,1), 1)],
            'z': [((0,0), 1), ((1,1), 1)]
        },
        # H2 = (A10+A11)B00 -> contributes to C10, C11
        {
            'x': [((1,0), 1), ((1,1), 1)],
            'y': [((0,0), 1)],
            'z': [((1,0), 1), ((1,1), -1)] # Wait, C11 = H1 - H2 ... so H2 has -1 in C11
        },
        # H3 = A00(B01-B11) -> contributes to C01, C11
        {
            'x': [((0,0), 1)],
            'y': [((0,1), 1), ((1,1), -1)],
            'z': [((0,1), 1), ((1,1), 1)] # C11 = ... + H3 ...
        },
        # H4 = A11(B10-B00) -> contributes to C00, C10
        {
            'x': [((1,1), 1)],
            'y': [((1,0), 1), ((0,0), -1)],
            'z': [((0,0), 1), ((1,0), 1)]
        },
        # H5 = (A00+A01)B11 -> contributes to C00, C01
        {
            'x': [((0,0), 1), ((0,1), 1)],
            'y': [((1,1), 1)],
            'z': [((0,0), -1), ((0,1), 1)] # C00 = ... - H5 ...
        },
        # H6 = (A10-A00)(B00+B01) -> contributes to C11
        {
            'x': [((1,0), 1), ((0,0), -1)],
            'y': [((0,0), 1), ((0,1), 1)],
            'z': [((1,1), 1)]
        },
        # H7 = (A01-A11)(B10+B11) -> contributes to C00
        {
            'x': [((0,1), 1), ((1,1), -1)],
            'y': [((1,0), 1), ((1,1), 1)],
            'z': [((0,0), 1)]
        }
    ]
    
    # Note: The 'z' coefficients in high_level_terms must match the reconstruction formula.
    # C00 = H1 + H4 - H5 + H7
    # C01 = H3 + H5
    # C10 = H2 + H4
    # C11 = H1 - H2 + H3 + H6
    
    # Let's verify my 'z' lists above:
    # H1: C00(+1), C11(+1). Correct.
    # H2: C10(+1), C11(-1). Correct.
    # H3: C01(+1), C11(+1). Correct.
    # H4: C00(+1), C10(+1). Correct.
    # H5: C00(-1), C01(+1). Correct.
    # H6: C11(+1). Correct.
    # H7: C00(+1). Correct.
    
    col = 0
    for hl_term in high_level_terms:
        # For this high-level term, we need to compute the product of two 2x2 block-matrices.
        # The "X" matrix is a 2x2 matrix where each element is a 2x2 block.
        # But Strassen works on scalar elements.
        # So we expand the high-level term into 7 low-level Strassen terms.
        
        # The high-level term defines linear combinations of blocks.
        # We need to map this to the 2x2 Strassen terms applied to the blocks.
        
        # Actually, it's easier to think of it this way:
        # The high-level term is a product P = X_block * Y_block.
        # X_block and Y_block are 2x2 matrices of 2x2 blocks? No.
        # X_block is a 2x2 matrix where each entry is a linear combo of A-blocks.
        # But Strassen is for 2x2 matrices.
        # So we treat the "entries" of the high-level X and Y as the "elements" of the 2x2 matrix for the low-level Strassen.
        
        # Let's refine the mapping.
        # High-level term h corresponds to a product M_h = X_h * Y_h.
        # X_h is a 2x2 matrix of 2x2 blocks.
        # Y_h is a 2x2 matrix of 2x2 blocks.
        # M_h is a 2x2 matrix of 2x2 blocks.
        
        # We decompose M_h using Strassen on the 2x2 structure of blocks.
        # This requires 7 products of 2x2 blocks.
        # Each product of 2x2 blocks is a 2x2x2 matrix multiplication, which we decompose with Strassen rank 7.
        
        # So, for each high-level term h:
        #   We have 7 "sub-terms" s1..s7 from the 2x2 Strassen on the block level.
        #   Each sub-term s_l involves:
        #     - A linear combination of 2x2 blocks from X_h (which are themselves linear combos of A-blocks).
        #     - A linear combination of 2x2 blocks from Y_h (which are themselves linear combos of B-blocks).
        #     - A contribution to the block result M_h.
        
        # This seems recursive.
        # Let's just generate the 49 terms directly by iterating:
        # 1. High-level Strassen term index h (0..6)
        # 2. Low-level Strassen term index s (0..6)
        
        # For a fixed h, we have:
        # X_h is defined by 'x' list of (block, coeff).
        # Y_h is defined by 'y' list of (block, coeff).
        
        # We need to form the 2x2 matrices X_h and Y_h (of blocks).
        # Then apply Strassen to these 2x2 matrices.
        
        # Let's build X_h and Y_h as 2x2 arrays of "block-linear-combinations".
        # A "block-linear-combination" is a dict {block_idx: coeff}.
        
        X_h = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), coeff in hl_term['x']:
            X_h[br][bc] = { (br, bc): coeff } # This is wrong. X_h[br][bc] should accumulate.
            
        # Let's re-initialize X_h properly.
        X_h = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), coeff in hl_term['x']:
            if (br, bc) in [(0,0), (0,1), (1,0), (1,1)]:
                # Wait, the 'x' list in high_level_terms contains the blocks that are summed.
                # e.g. H1: A00 + A11.
                # This means the "X" matrix for Strassen is not a single matrix, but the sum?
                # No, Strassen term H1 = (A00+A11)(B00+B11).
                # This is a product of two sums.
                # (A00+A11) is not a 2x2 matrix. It's a sum of two 2x2 matrices.
                # So H1 is not a single 2x2 block product.
                # It is (A00+A11) * (B00+B11).
                # Let X_sum = A00 + A11.
                # Let Y_sum = B00 + B11.
                # H1 = X_sum * Y_sum.
                # X_sum is a 2x2 matrix. Y_sum is a 2x2 matrix.
                # So we can apply Strassen to X_sum and Y_sum?
                # No, Strassen applies to the 2x2 structure of the blocks.
                # Here X_sum is just a 2x2 matrix of scalars (if we view blocks as scalars)?
                # No, X_sum is a 2x2 matrix where each element is a 2x2 block.
                # Wait, A00 is a 2x2 block. A11 is a 2x2 block.
                # A00 + A11 is a 2x2 matrix of blocks? No, it's a single 2x2 matrix formed by adding two 2x2 blocks.
                # So X_sum is a 2x2 matrix of scalars? No, it's a 2x2 matrix of "block values".
                # Let's clarify dimensions.
                # A is 4x4.
                # A00 is 2x2.
                # A00 + A11 is a 2x2 matrix.
                # So X_sum is a 2x2 matrix.
                # Y_sum is a 2x2 matrix.
                # Product H1 = X_sum * Y_sum is a 2x2 matrix.
                # This 2x2 matrix is a block of C.
                # So H1 is a 2x2 block of C.
                # To compute H1 = X_sum * Y_sum, we can use Strassen on the 2x2 matrices X_sum and Y_sum?
                # No, X_sum and Y_sum are 2x2 matrices of scalars?
                # No, the elements of X_sum are not scalars, they are entries of the 2x2 blocks?
                # No, X_sum is a 2x2 matrix. Its entries are the sums of corresponding entries of A00 and A11.
                # So X_sum is a 2x2 matrix of scalars (relative to the block level).
                # Wait, if X_sum is 2x2, and Y_sum is 2x2, their product is 2x2.
                # This product is one block of C (e.g. C00).
                # But C00 is a 2x2 block of the final 4x4 matrix C.
                # So H1 is a 2x2 matrix.
                # To compute the 2x2 matrix product H1 = X_sum * Y_sum, we can use Strassen rank 7.
                # Yes!
                # So for each high-level term, we have a 2x2 matrix product.
                # We decompose that 2x2 product using Strassen rank 7.
                # Total 7 * 7 = 49 terms.
                
                pass
        
        # So, for each high-level term h:
        # 1. Construct X_h (2x2 matrix) and Y_h (2x2 matrix).
        #    X_h[br][bc] is a linear combination of A-blocks?
        #    No, X_h is a 2x2 matrix where the entries are linear combinations of the *elements* of the blocks?
        #    Let's look at H1 = (A00+A11)(B00+B11).
        #    A00 is 2x2. A11 is 2x2.
        #    A00+A11 is 2x2.
        #    So X_h is a 2x2 matrix.
        #    Its entries are (A00[0,0] + A11[0,0]), etc.
        #    So X_h[0,0] = A00[0,0] + A11[0,0].
        #    This is a linear combination of specific elements of A.
        
        # Let's define X_h and Y_h as 2x2 arrays of "element-linear-combinations".
        # An "element-linear-combination" is a dict {global_a_idx: coeff}.
        
        X_h = [[{} for _ in range(2)] for _ in range(2)]
        Y_h = [[{} for _ in range(2)] for _ in range(2)]
        
        # Fill X_h
        for (br, bc), coeff in hl_term['x']:
            # Add block (br, bc) to X_h
            # The block (br, bc) occupies rows br*2, br*2+1 and cols bc*2, bc*2+1 in A.
            # But X_h is a 2x2 matrix.
            # The entry X_h[0,0] corresponds to the top-left element of the resulting 2x2 matrix.
            # The resulting 2x2 matrix is the sum of the blocks.
            # So X_h[0,0] += A[br*2][bc*2] * coeff.
            # X_h[0,1] += A[br*2][bc*2+1] * coeff.
            # X_h[1,0] += A[br*2+1][bc*2] * coeff.
            # X_h[1,1] += A[br*2+1][bc*2+1] * coeff.
            
            for lr in range(2):
                for lc in range(2):
                    global_row = br * 2 + lr
                    global_col = bc * 2 + lc
                    global_a_idx = global_row * 4 + global_col
                    X_h[lr][lc][global_a_idx] = X_h[lr][lc].get(global_a_idx, 0) + coeff
                    
        # Fill Y_h
        for (br, bc), coeff in hl_term['y']:
            for lr in range(2):
                for lc in range(2):
                    global_row = br * 2 + lr
                    global_col = bc * 2 + lc
                    global_b_idx = global_row * 9 + global_col
                    Y_h[lr][lc][global_b_idx] = Y_h[lr][lc].get(global_b_idx, 0) + coeff
        
        # Now we have X_h and Y_h as 2x2 matrices of "element-linear-combinations".
        # We apply Strassen to compute Z_h = X_h * Y_h.
        # Z_h will be a 2x2 matrix of "element-linear-combinations" for C.
        
        # Strassen terms for 2x2:
        # s1 = (X00+X11)(Y00+Y11)
        # ...
        
        # We iterate over the 7 low-level Strassen terms.
        for s_term in strassen_2x2:
            # Construct X_s and Y_s (vectors of element-linear-combinations)
            # X_s is a linear combination of X_h entries.
            # Y_s is a linear combination of Y_h entries.
            
            X_s = {}
            for (idx, coeff) in s_term['x']:
                # idx is 0..3 for 2x2.
                r, c = idx // 2, idx % 2
                # Add X_h[r][c] * coeff to X_s
                for a_idx, val in X_h[r][c].items():
                    X_s[a_idx] = X_s.get(a_idx, 0) + val * coeff
                    
            Y_s = {}
            for (idx, coeff) in s_term['y']:
                r, c = idx // 2, idx % 2
                for b_idx, val in Y_h[r][c].items():
                    Y_s[b_idx] = Y_s.get(b_idx, 0) + val * coeff
            
            # Now we have X_s and Y_s.
            # This term contributes to Z_h.
            # The contribution is defined by s_term['z'].
            # For each (z_idx, z_coeff) in s_term['z']:
            #   Add X_s * Y_s * z_coeff to Z_h[z_idx].
            
            for (z_idx, z_coeff) in s_term['z']:
                z_r, z_c = z_idx // 2, z_idx % 2
                # Z_h[z_r][z_c] accumulates this term.
                # The term is X_s * Y_s.
                # This is a rank-1 term in the tensor decomposition?
                # No, X_s and Y_s are linear combinations.
                # The term is:
                # sum_{a in X_s} sum_{b in Y_s} X_s[a] * Y_s[b] * z_coeff
                # This adds to the block at (z_r, z_c) of the high-level result.
                # The block (z_r, z_c) of the high-level result corresponds to a 2x2 block of C.
                # Wait, Z_h is the result of the 2x2 Strassen.
                # Z_h is a 2x2 matrix.
                # Each entry Z_h[z_r][z_c] is a linear combination of C-blocks?
                # No, Z_h is the 2x2 matrix product X_h * Y_h.
                # So Z_h[z_r][z_c] is the (z_r, z_c) entry of the product.
                # This entry is a scalar (in the context of the block-level Strassen).
                # But it is formed from products of elements of X_h and Y_h.
                # So it is a linear combination of products of A-elements and B-elements.
                # This linear combination contributes to the C-block.
                
                # Which C-block?
                # The high-level term h contributes to specific C-blocks.
                # hl_term['z'] lists the C-blocks and their coefficients.
                # For each (c_br, c_bc), c_coeff in hl_term['z']:
                #   The value Z_h[z_r][z_c] * c_coeff is added to the C-block (c_br, c_bc) at position (z_r, z_c).
                
                for (c_br, c_bc), c_coeff in hl_term['z']:
                    # The C-block (c_br, c_bc) is a 2x2 block.
                    # The entry (z_r, z_c) of this block is:
                    # global_c_row = c_br * 2 + z_r
                    # global_c_col = c_bc * 2 + z_c
                    # global_w_idx = global_c_col * 4 + global_c_row
                    
                    total_coeff = z_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    
                    # Add the term to U, V, W
                    # U: X_s
                    # V: Y_s
                    # W: 1 at global_w_idx
                    
                    # We need to add a term for each pair (a_idx, b_idx) in X_s, Y_s.
                    # Term: U[a_idx] += X_s[a_idx] * total_coeff
                    #       V[b_idx] += Y_s[b_idx]
                    #       W[global_w_idx] += 1
                    # Wait, this would create many terms.
                    # But X_s and Y_s are sparse (few non-zero entries).
                    # Actually, we can just add the whole linear combination as a single "term" in the decomposition?
                    # No, the decomposition requires rank-1 terms.
                    # So we must expand X_s and Y_s into individual elements.
                    
                    for a_idx, a_val in X_s.items():
                        if a_val == 0: continue
                        for b_idx, b_val in Y_s.items():
                            if b_val == 0: continue
                            # Term:
                            # U[a_idx] += a_val
                            # V[b_idx] += b_val
                            # W[global_w_idx] += total_coeff
                            # But we need to distribute the coefficient.
                            # The term is (a_val * b_val * total_coeff) * (U_e_a * V_e_b * W_e_c).
                            # We can set:
                            # U[a_idx] += a_val
                            # V[b_idx] += b_val
                            # W[global_w_idx] += total_coeff
                            # This works if we sum over all such terms.
                            
                            U[a_idx][col] += a_val
                            V[b_idx][col] += b_val
                            W[global_w_idx][col] += total_coeff
                            
                    col += 1
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
