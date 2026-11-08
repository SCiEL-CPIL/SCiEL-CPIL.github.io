import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-53 decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Split B into B1 (4x4) and B2 (4x1)
    - C = A*B1 + A*B2
    - A*B2: 4 terms (schoolbook)
    - A*B1: Use 2x2 block Strassen with 2x2 scalar Strassen inside
      Each 2x2 block product needs 7 scalar multiplications
      7 block-level terms * 7 scalar multiplications = 49 scalar multiplications
    - Total rank = 49 + 4 = 53
    """
    n, m, p = 4, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # We'll build U, V, W as lists of lists
    # U: shape (n*m, rank) = (16, 53)
    # V: shape (m*p, rank) = (20, 53)
    # W: shape (p*n, rank) = (20, 53)
    
    rank = 53
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(20)]
    W = [[0.0] * rank for _ in range(20)]
    
    # Helper: 2x2 Strassen for matrices represented as 4 elements [a00, a01, a10, a11]
    # Returns 7 products and coefficients to reconstruct [c00, c01, c10, c11]
    # For X * Y where X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)(y00)
    # M3 = (x00)(y01 - y11)
    # M4 = (x11)(y10 - y00)
    # M5 = (x00 + x01)(y11)
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    # C = [[M1 + M4 - M5 + M7, M3 + M5],
    #      [M2 + M4, M1 - M2 + M3 + M6]]
    
    # Define the 7 Strassen products for 2x2 case
    # Each product is (sum of X elements) * (sum of Y elements)
    # X elements: [x00, x01, x10, x11] with coefficients
    # Y elements: [y00, y01, y10, y11] with coefficients
    
    strassen_products = [
        # M1: (x00 + x11)(y00 + y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (x10 + x11)(y00)
        ({2: 1, 3: 1}, {0: 1}),
        # M3: (x00)(y01 - y11)
        ({0: 1}, {1: 1, 3: -1}),
        # M4: (x11)(y10 - y00)
        ({3: 1}, {2: 1, 0: -1}),
        # M5: (x00 + x01)(y11)
        ({0: 1, 1: 1}, {3: 1}),
        # M6: (x10 - x00)(y00 + y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7: (x01 - x11)(y10 + y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # Coefficients for reconstructing C from M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_coeffs = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},   # C00
        1: {2: 1, 4: 1},                 # C01
        2: {1: 1, 3: 1},                 # C10
        3: {0: 1, 1: -1, 2: 1, 5: 1},   # C11
    }
    
    # Block structure for 4x4 * 4x4:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # For each of the 7 block-level Strassen products, we need to compute a 2x2 * 2x2 product
    # using scalar Strassen, giving 7 scalar multiplications per block product
    
    # Mapping from A row/col to index in U (n*m = 16)
    # A is 4x4, so A[i,j] is at index i*4 + j in U
    
    # Mapping from B row/col to index in V (m*p = 20 for full B, but B1 is 4x4)
    # B1 is 4x4, so B1[i,j] is at index i*4 + j in V (for j=0..3)
    
    # Mapping from C row/col to index in W (p*n = 20)
    # C is 4x5, so C[i,j] is at index j*4 + i in W (for j=0..4)
    
    col = 0
    
    # For each of the 7 block-level Strassen products:
    for block_prod_idx in range(7):
        # Determine which blocks X and Y are involved
        if block_prod_idx == 0:  # M1: (A00 + A11)(B00 + B11)
            x_blocks = [(0, 0, 1), (2, 2, 1)]  # (row_start, col_start, sign)
            y_blocks = [(0, 0, 1), (2, 2, 1)]
        elif block_prod_idx == 1:  # M2: (A10 + A11)(B00)
            x_blocks = [(2, 0, 1), (2, 2, 1)]
            y_blocks = [(0, 0, 1)]
        elif block_prod_idx == 2:  # M3: (A00)(B01 - B11)
            x_blocks = [(0, 0, 1)]
            y_blocks = [(0, 2, 1), (2, 2, -1)]
        elif block_prod_idx == 3:  # M4: (A11)(B10 - B00)
            x_blocks = [(2, 2, 1)]
            y_blocks = [(2, 0, 1), (0, 0, -1)]
        elif block_prod_idx == 4:  # M5: (A00 + A01)(B11)
            x_blocks = [(0, 0, 1), (0, 2, 1)]
            y_blocks = [(2, 2, 1)]
        elif block_prod_idx == 5:  # M6: (A10 - A00)(B00 + B01)
            x_blocks = [(2, 0, 1), (0, 0, -1)]
            y_blocks = [(0, 0, 1), (0, 2, 1)]
        elif block_prod_idx == 6:  # M7: (A01 - A11)(B10 + B11)
            x_blocks = [(0, 2, 1), (2, 2, -1)]
            y_blocks = [(2, 0, 1), (2, 2, 1)]
        
        # For each element in the 2x2 X and Y blocks, we need to build the scalar products
        # X is a 2x2 matrix formed by summing the specified blocks with signs
        # Y is a 2x2 matrix formed by summing the specified blocks with signs
        
        # Build X coefficients for each of the 4 elements [x00, x01, x10, x11]
        # X[r, c] = sum of sign * A[block_row + r, block_col + c]
        x_coeff = [[0.0] * 4 for _ in range(4)]  # x_coeff[i][j] = coefficient for A[i,j] in X[r,c]
        # Actually, let's think of it differently.
        # X is 2x2. X[0,0] = sum of A[block_row+0, block_col+0] with signs
        # X[0,1] = sum of A[block_row+0, block_col+1] with signs
        # X[1,0] = sum of A[block_row+1, block_col+0] with signs
        # X[1,1] = sum of A[block_row+1, block_col+1] with signs
        
        # For each X element (r, c) in 0..1, 0..1:
        # x_coeff[r][c] is a dict mapping A indices to coefficients
        x_coeff = [{} for _ in range(4)]  # for x00, x01, x10, x11
        for block_row, block_col, sign in x_blocks:
            for r in range(2):
                for c in range(2):
                    idx = r * 2 + c  # 0=x00, 1=x01, 2=x10, 3=x11
                    a_i = block_row + r
                    a_j = block_col + c
                    a_idx = a_i * 4 + a_j
                    x_coeff[idx][a_idx] = x_coeff[idx].get(a_idx, 0.0) + sign
        
        # Similarly for Y
        y_coeff = [{} for _ in range(4)]
        for block_row, block_col, sign in y_blocks:
            for r in range(2):
                for c in range(2):
                    idx = r * 2 + c
                    b_i = block_row + r
                    b_j = block_col + c
                    b_idx = b_i * 4 + b_j  # For B1 (4x4), index is i*4+j
                    y_coeff[idx][b_idx] = y_coeff[idx].get(b_idx, 0.0) + sign
        
        # Now for each of the 7 scalar Strassen products:
        for scalar_prod_idx in range(7):
            x_strassen_coeff, y_strassen_coeff = strassen_products[scalar_prod_idx]
            
            # Build the linear combination for X
            x_comb = {}
            for x_idx, x_sign in x_strassen_coeff.items():
                for a_idx, coeff in x_coeff[x_idx].items():
                    x_comb[a_idx] = x_comb.get(a_idx, 0.0) + x_sign * coeff
            
            # Build the linear combination for Y
            y_comb = {}
            for y_idx, y_sign in y_strassen_coeff.items():
                for b_idx, coeff in y_coeff[y_idx].items():
                    y_comb[b_idx] = y_comb.get(b_idx, 0.0) + y_sign * coeff
            
            # Add to U, V, W
            for a_idx, coeff in x_comb.items():
                if coeff != 0:
                    U[a_idx][col] += coeff
            for b_idx, coeff in y_comb.items():
                if coeff != 0:
                    V[b_idx][col] += coeff
            
            col += 1
    
    # Now we need to fill in W for the C matrix
    # C = A * B1 + A * B2
    # First part: A * B1 using block Strassen
    # C_block = [[C00, C01], [C10, C11]] where each is 2x2
    # C_block = M1_block + M4_block - M5_block + M7_block (for C00)
    # etc.
    
    # For each of the 7 block-level products, we have 7 scalar products
    # The scalar products contribute to the 2x2 block result
    # We need to map these to the final C entries
    
    # Let's redo the W construction more carefully
    # Reset W for the first 49 columns
    for i in range(20):
        for j in range(49):
            W[i][j] = 0.0
    
    col = 0
    for block_prod_idx in range(7):
        # Determine which blocks X and Y are involved (same as above)
        if block_prod_idx == 0:  # M1: (A00 + A11)(B00 + B11)
            x_blocks = [(0, 0, 1), (2, 2, 1)]
            y_blocks = [(0, 0, 1), (2, 2, 1)]
        elif block_prod_idx == 1:  # M2: (A10 + A11)(B00)
            x_blocks = [(2, 0, 1), (2, 2, 1)]
            y_blocks = [(0, 0, 1)]
        elif block_prod_idx == 2:  # M3: (A00)(B01 - B11)
            x_blocks = [(0, 0, 1)]
            y_blocks = [(0, 2, 1), (2, 2, -1)]
        elif block_prod_idx == 3:  # M4: (A11)(B10 - B00)
            x_blocks = [(2, 2, 1)]
            y_blocks = [(2, 0, 1), (0, 0, -1)]
        elif block_prod_idx == 4:  # M5: (A00 + A01)(B11)
            x_blocks = [(0, 0, 1), (0, 2, 1)]
            y_blocks = [(2, 2, 1)]
        elif block_prod_idx == 5:  # M6: (A10 - A00)(B00 + B01)
            x_blocks = [(2, 0, 1), (0, 0, -1)]
            y_blocks = [(0, 0, 1), (0, 2, 1)]
        elif block_prod_idx == 6:  # M7: (A01 - A11)(B10 + B11)
            x_blocks = [(0, 2, 1), (2, 2, -1)]
            y_blocks = [(2, 0, 1), (2, 2, 1)]
        
        for scalar_prod_idx in range(7):
            # This scalar product contributes to the block result
            # We need to determine which C entries it affects and with what coefficient
            
            # The scalar product is M_scalar = (linear_comb_of_X) * (linear_comb_of_Y)
            # This M_scalar contributes to the 2x2 block result via c_coeffs
            # The 2x2 block result then contributes to specific C entries
            
            # For block_prod_idx, the block result is a 2x2 matrix
            # The position of this block in C depends on block_prod_idx
            # Actually, all 7 block products contribute to the same C structure:
            # C = M1_block + M4_block - M5_block + M7_block (for C00 position)
            # etc.
            
            # The block result at position (br, bc) in the 2x2 block grid
            # contributes to C entries at rows br*2..br*2+1 and cols bc*2..bc*2+1
            
            # For block_prod_idx, we need to know which block position it corresponds to
            # and how it combines with others
            
            # Actually, the 7 block products are:
            # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
            # M2 = (A10+A11)(B00) -> contributes to C10, C11
            # M3 = (A00)(B01-B11) -> contributes to C01, C11
            # M4 = (A11)(B10-B00) -> contributes to C00, C10
            # M5 = (A00+A01)(B11) -> contributes to C00, C01
            # M6 = (A10-A00)(B00+B01) -> contributes to C11
            # M7 = (A01-A11)(B10+B11) -> contributes to C00
            
            # Each block product is a 2x2 matrix, so it has 4 entries
            # Each entry is a linear combination of the 7 scalar products
            
            # Let's compute the contribution of scalar_prod_idx to each of the 4 block entries
            # Then map those to C entries
            
            # For the block result, entry (br, bc) where br, bc in {0,1}:
            # This corresponds to C rows br*2, br*2+1 and cols bc*2, bc*2+1
            
            # The coefficient of M_scalar in block entry (br, bc) is given by c_coeffs
            # But c_coeffs is for the 2x2 case where M1..M7 are the 7 products
            
            # So for block_prod_idx, the contribution to block entry (br, bc) is:
            # c_coeffs[br*2+bc][scalar_prod_idx]
            
            # Then this block entry contributes to C entries at:
            # C[br*2 + r, bc*2 + c] for r, c in {0,1}
            # But wait, the block entry (br, bc) IS a 2x2 matrix? No.
            
            # Let me reclarify:
            # A is 4x4, split into 2x2 blocks of 2x2 matrices
            # C is 4x4, split into 2x2 blocks of 2x2 matrices
            # Each block C[br, bc] is a 2x2 matrix
            # C[br, bc] = sum of contributions from M1..M7 block products
            
            # For each block product Mk (which is a 2x2 matrix), its entry (r, c) 
            # contributes to C[br*2+r, bc*2+c]
            
            # So for block_prod_idx, we need to know which block position (br, bc) it corresponds to
            # Actually, all 7 block products contribute to ALL 4 block positions? No.
            
            # Let's look at the formulas:
            # C00_block = M1_block + M4_block - M5_block + M7_block
            # C01_block = M3_block + M5_block
            # C10_block = M2_block + M4_block
            # C11_block = M1_block - M2_block + M3_block + M6_block
            
            # So M1_block contributes to C00_block and C11_block
            # M2_block contributes to C10_block and C11_block
            # etc.
            
            block_positions = {
                0: [(0, 0, 1), (1, 1, 1)],   # M1
                1: [(1, 0, 1), (1, 1, -1)],  # M2
                2: [(0, 1, 1), (1, 1, 1)],   # M3
                3: [(0, 0, 1), (1, 0, 1)],   # M4
                4: [(0, 0, -1), (0, 1, 1)],  # M5
                5: [(1, 1, 1)],               # M6
                6: [(0, 0, 1)],               # M7
            }
            
            for br, bc, sign in block_positions[block_prod_idx]:
                # This block product contributes to block (br, bc) with sign
                # The block entry (r, c) of this 2x2 matrix contributes to C[br*2+r, bc*2+c]
                # The coefficient of scalar_prod_idx in block entry (r, c) is c_coeffs[r*2+c][scalar_prod_idx]
                for r in range(2):
                    for c in range(2):
                        coeff = c_coeffs[r * 2 + c].get(scalar_prod_idx, 0.0)
                        if coeff != 0:
                            c_i = br * 2 + r
                            c_j = bc * 2 + c
                            # C index in W: j*4 + i
                            w_idx = c_j * 4 + c_i
                            W[w_idx][col] += sign * coeff
            
            col += 1
    
    # Now add the 4 terms for A * B2 (the 5th column of B)
    # B2 is 4x1, so A*B2 is 4x1
    # Schoolbook: C[i, 4] = sum_j A[i, j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1.0
            V[j * 4 + 4][col] = 1.0  # B index: j*4+4 for the 5th column
            W[4 * 4 + i][col] = 1.0  # C index: 4*4+i for column 4
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
