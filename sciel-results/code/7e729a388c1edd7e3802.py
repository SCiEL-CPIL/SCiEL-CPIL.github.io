import time
import random

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    a tensor-product / block-based construction derived from Strassen's
    2x2 algorithm (rank 7) applied to the 4x4 problem.
    
    The 4x4 matrices are viewed as 2x2 block matrices of 2x2 blocks.
    Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]].
    The product C = A*B has blocks C11, C12, C21, C22.
    
    Strassen's 7 products for 2x2 blocks (where each block is 2x2):
    P1 = (A11 + A22) * (B11 + B22)
    P2 = (A21 + A22) * B11
    P3 = A11 * (B12 - B22)
    P4 = A22 * (B21 - B11)
    P5 = (A11 + A12) * B22
    P6 = (A21 - A11) * (B11 + B12)
    P7 = (A12 - A22) * (B21 + B22)
    
    Output blocks:
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    Each Pi is a product of two 2x2 matrices.
    Using Strassen's algorithm on the 2x2 blocks gives 7 scalar products per block product.
    Total rank = 7 (block products) * 7 (scalar products per block) = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 formulas for 2x2 matmul:
    # Given X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # s1 = (x0+x3)*(y0+y3)
    # s2 = (x2+x3)*y0
    # s3 = x0*(y1-y3)
    # s4 = x3*(y2-y0)
    # s5 = (x0+x1)*y3
    # s6 = (x2-x0)*(y0+y1)
    # s7 = (x1-x3)*(y2+y3)
    # Result:
    # [[s1+s4-s5+s7, s3+s5],
    #  [s2+s4, s1-s2+s3+s6]]
    
    # We define the 7 block-level linear combinations for A and B.
    # Each combination is a 2x2 matrix of coefficients applied to the 2x2 blocks.
    # Block (i, j) corresponds to rows 2i..2i+1 and cols 2j..2j+1.
    
    # A-side combinations (coefficients for A11, A12, A21, A22)
    # Format: list of 7 dicts, each dict maps block_idx (0=A11, 1=A12, 2=A21, 3=A22) to coeff
    A_combos = [
        {0: 1, 3: 1},       # A11 + A22
        {2: 1, 3: 1},       # A21 + A22
        {0: 1},             # A11
        {3: 1},             # A22
        {0: 1, 1: 1},       # A11 + A12
        {2: 1, 0: -1},      # A21 - A11
        {1: 1, 3: -1},      # A12 - A22
    ]
    
    # B-side combinations (coefficients for B11, B12, B21, B22)
    B_combos = [
        {0: 1, 3: 1},       # B11 + B22
        {0: 1},             # B11
        {1: 1, 3: -1},      # B12 - B22
        {2: 1, 0: -1},      # B21 - B11
        {3: 1},             # B22
        {0: 1, 1: 1},       # B11 + B12
        {2: 1, 3: 1},       # B21 + B22
    ]
    
    # Strassen scalar coefficients for each of the 7 sub-products (s1..s7)
    # for each of the 4 output blocks of the 2x2 result (r,c) in {0,1}x{0,1}
    # strassen_coeffs[sub_prod_idx][row][col]
    strassen_coeffs = [
        # s1
        [[1, 0], [0, 1]],
        # s2
        [[0, 0], [1, -1]],
        # s3
        [[0, 1], [0, 1]],
        # s4
        [[1, 0], [1, 0]],
        # s5
        [[-1, 1], [0, 0]],
        # s6
        [[0, 0], [0, 1]],
        # s7
        [[1, 0], [0, 0]],
    ]
    
    # Block contributions for the 4x4 result
    # block_contrib[block_prod_idx][block_row][block_col]
    # block_prod_idx is 0..6 (P1..P7)
    # block_row, block_col are 0..1 for the 2x2 block structure of C
    block_contrib = [
        # P1
        [[1, 0], [0, 1]],
        # P2
        [[0, 0], [1, -1]],
        # P3
        [[0, 1], [0, 1]],
        # P4
        [[1, 0], [1, 0]],
        # P5
        [[-1, 1], [0, 0]],
        # P6
        [[0, 0], [0, 1]],
        # P7
        [[1, 0], [0, 0]],
    ]
    
    col = 0
    for bp_idx in range(7):
        A_combo = A_combos[bp_idx]
        B_combo = B_combos[bp_idx]
        
        # For each of the 7 Strassen sub-products within this block product
        for sp_idx in range(7):
            # Determine the 2x2 matrices of coefficients for the A-block and B-block
            # We need to expand the block-level coefficients to element-level (4x4)
            
            # A_element_coeffs: 4x4 matrix of coefficients for U
            # B_element_coeffs: 4x4 matrix of coefficients for V
            
            # The block coefficients apply to 2x2 sub-blocks.
            # We need to combine the block coefficients with the Strassen scalar coefficients.
            
            # Let's build the U and V vectors for this column (rank index)
            # U is size 16 (4x4), V is size 16 (4x4), W is size 16 (4x4)
            
            u_vals = [0] * 16
            v_vals = [0] * 16
            w_vals = [0] * 16
            
            # Iterate over the 2x2 blocks of A
            for br in range(2):
                for bc in range(2):
                    block_idx = br * 2 + bc
                    coeff_block = A_combo.get(block_idx, 0)
                    if coeff_block == 0:
                        continue
                    
                    # The block is 2x2. We need to apply Strassen's s-sp_idx coefficients to the elements within this block?
                    # NO. The structure is:
                    # The "block product" P_bp is a 2x2 matrix product of two 2x2 matrices.
                    # Let X_block be the 2x2 matrix formed by the A blocks.
                    # Let Y_block be the 2x2 matrix formed by the B blocks.
                    # P_bp = X_block * Y_block.
                    # We use Strassen to compute X_block * Y_block using 7 scalar products.
                    # Each scalar product s_sp is a product of linear combinations of elements of X_block and Y_block.
                    
                    # So, for a fixed bp_idx and sp_idx:
                    # We have linear forms L_A(X_block) and L_B(Y_block).
                    # L_A is a linear combination of the 4 blocks of X_block.
                    # L_B is a linear combination of the 4 blocks of Y_block.
                    # The product is L_A * L_B.
                    # This product contributes to the entries of P_bp.
                    
                    # Wait, the Strassen algorithm computes the product of two 2x2 matrices.
                    # The inputs are 2x2 matrices.
                    # In our case, the "entries" of the 2x2 matrices are 2x2 blocks.
                    # So, the "scalar" product in Strassen is actually a block product (2x2 * 2x2).
                    # To get a rank-1 decomposition, we need to decompose the block product into scalar products.
                    # The block product (2x2 * 2x2) has rank 4 (schoolbook) or 7 (Strassen).
                    # The current code uses Strassen for the blocks too?
                    # Yes, the rank 49 comes from 7 (outer Strassen) * 7 (inner Strassen).
                    
                    # So, for each outer Strassen product (bp_idx), we have a term L_A_block * L_B_block.
                    # This term is a 2x2 block product.
                    # We decompose this 2x2 block product using inner Strassen (sp_idx).
                    
                    # So, for a fixed bp_idx and sp_idx:
                    # We need the linear forms for the inner Strassen.
                    
                    # Inner Strassen s_sp_idx uses:
                    # x0, x1, x2, x3 from the 2x2 "matrix" of blocks.
                    # y0, y1, y2, y3 from the 2x2 "matrix" of blocks.
                    
                    # The coefficients for x0..x3 are determined by sp_idx.
                    # The coefficients for y0..y3 are determined by sp_idx.
                    
                    # Let's define the inner Strassen coefficients for the 4 "entries" (blocks)
                    # inner_x_coeffs[sp_idx] = [c0, c1, c2, c3]
                    # inner_y_coeffs[sp_idx] = [c0, c1, c2, c3]
                    
                    inner_x_coeffs = [
                        [1, 0, 0, 1], # s1: x0+x3
                        [0, 0, 1, 1], # s2: x2+x3
                        [1, 0, 0, 0], # s3: x0
                        [0, 0, 0, 1], # s4: x3
                        [1, 1, 0, 0], # s5: x0+x1
                        [0, 0, 1, -1],# s6: x2-x0
                        [0, 1, 0, -1] # s7: x1-x3
                    ]
                    
                    inner_y_coeffs = [
                        [1, 0, 0, 1], # s1: y0+y3
                        [1, 0, 0, 0], # s2: y0
                        [0, 1, 0, -1],# s3: y1-y3
                        [0, 0, 1, -1],# s4: y2-y0
                        [0, 0, 0, 1], # s5: y3
                        [1, 1, 0, 0], # s6: y0+y1
                        [0, 0, 1, 1]  # s7: y2+y3
                    ]
                    
                    cx = inner_x_coeffs[sp_idx]
                    cy = inner_y_coeffs[sp_idx]
                    
                    # Now, for each block position (br, bc) in the 2x2 block matrix:
                    # The coefficient for block (br, bc) in L_A is A_combo.get(br*2+bc, 0) * cx[br*2+bc]
                    # The coefficient for block (br, bc) in L_B is B_combo.get(br*2+bc, 0) * cy[br*2+bc]
                    
                    # We add these to u_vals and v_vals.
                    # u_vals corresponds to the flat 4x4 A matrix.
                    # v_vals corresponds to the flat 4x4 B matrix.
                    
                    for br in range(2):
                        for bc in range(2):
                            block_idx = br * 2 + bc
                            coeff_a = A_combo.get(block_idx, 0) * cx[block_idx]
                            coeff_b = B_combo.get(block_idx, 0) * cy[block_idx]
                            
                            if coeff_a != 0 or coeff_b != 0:
                                # The block is 2x2.
                                # Rows in 4x4: 2*br, 2*br+1
                                # Cols in 4x4: 2*bc, 2*bc+1
                                for r in range(2):
                                    for c in range(2):
                                        row = 2 * br + r
                                        col_idx = 2 * bc + c
                                        flat_idx = row * 4 + col_idx
                                        
                                        u_vals[flat_idx] += coeff_a
                                        v_vals[flat_idx] += coeff_b
                                        
            # Now we have u_vals and v_vals for this rank index.
            # We need to compute w_vals.
            # The product s_sp contributes to the output blocks of P_bp.
            # The contribution of s_sp to the 2x2 block result R_block is given by strassen_coeffs[sp_idx].
            # R_block is a 2x2 matrix of 2x2 blocks.
            # The entry (r, c) of R_block is a 2x2 block.
            # The coefficient for s_sp in R_block[r, c] is strassen_coeffs[sp_idx][r][c].
            
            # This R_block is the result of the block product P_bp.
            # The block product P_bp contributes to the final 4x4 result C via block_contrib[bp_idx].
            # C_block[r, c] = sum_bp block_contrib[bp_idx][r][c] * P_bp[r, c]
            
            # So, for each block position (br, bc) in the 2x2 block structure:
            # The contribution to C_block[br, bc] from s_sp is:
            # coeff = strassen_coeffs[sp_idx][br][bc] * block_contrib[bp_idx][br][bc]
            
            # C_block[br, bc] is a 2x2 block.
            # The elements of this block are filled with this coefficient.
            
            for br in range(2):
                for bc in range(2):
                    coeff = strassen_coeffs[sp_idx][br][bc] * block_contrib[bp_idx][br][bc]
                    if coeff != 0:
                        for r in range(2):
                            for c in range(2):
                                row = 2 * br + r
                                col_idx = 2 * bc + c
                                # W index: k*n + i. Here k is col, i is row.
                                # W is (p*n, rank) = (16, rank).
                                # Index = col_idx * 4 + row
                                flat_idx = col_idx * 4 + row
                                w_vals[flat_idx] += coeff
            
            # Assign to U, V, W
            for i in range(16):
                U[i][col] = u_vals[i]
                V[i][col] = v_vals[i]
                W[i][col] = w_vals[i]
            
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try the rank-49 Strassen-based decomposition
    U, V, W = _strassen_4x4()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 64 if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
