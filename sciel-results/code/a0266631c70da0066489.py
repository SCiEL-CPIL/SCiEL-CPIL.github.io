"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We split B (4x8) into two 4x4 blocks: B_L = B[:, :4], B_R = B[:, 4:]
    # C = A * B = [A*B_L | A*B_R]
    # Each A*B_block is a 4x4 * 4x4 multiplication.
    # We use Strassen's algorithm for 4x4 * 4x4, which can be done with rank 49
    # (using Strassen at the 2x2 level).
    # Total rank = 49 + 49 = 98.
    
    # Strassen 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen 4x4 * 4x4 (using 2x2 blocks):
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # We compute 7 products of 2x2 matrices, each using Strassen (7 mults) = 49 total
    
    # Let's define the 4x4 Strassen in terms of 2x2 block indices.
    # A block (i,j) where i,j in {0,1} refers to 2x2 block.
    # Within a 2x2 block, indices (r,c) where r,c in {0,1}.
    
    # For 4x4 * 4x4:
    # M1_4x4 = (A11 + A22) * (B11 + B22)
    # M2_4x4 = (A21 + A22) * B11
    # M3_4x4 = A11 * (B12 - B22)
    # M4_4x4 = A22 * (B21 - B11)
    # M5_4x4 = (A11 + A12) * B22
    # M6_4x4 = (A21 - A11) * (B11 + B12)
    # M7_4x4 = (A12 - A22) * (B21 + B22)
    
    # Each of these is a 2x2 * 2x2 multiplication, done via Strassen (7 products).
    # So 7 * 7 = 49 products total for 4x4 * 4x4.
    
    # For each product M_k_4x4 = X * Y where X, Y are 2x2:
    # We decompose X*Y into 7 scalar products using Strassen 2x2.
    # Then the contribution to C blocks is determined by how M_k_4x4 appears in C.
    
    # Let's build the decomposition step by step.
    
    # First, define the 2x2 Strassen decomposition.
    # For X * Y = Z where X, Y, Z are 2x2:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # N1 = (x00+x11)(y00+y11)
    # N2 = (x10+x11)y00
    # N3 = x00(y01-y11)
    # N4 = x11(y10-y00)
    # N5 = (x00+x01)y11
    # N6 = (x10-x00)(y00+y01)
    # N7 = (x01-x11)(y10+y11)
    # z00 = N1 + N4 - N5 + N7
    # z01 = N3 + N5
    # z10 = N2 + N4
    # z11 = N1 - N2 + N3 + N6
    
    # Each N_i is a product of two linear combinations.
    # We need to express each N_i as sum of scalar products a*b.
    
    # Let's define a helper that, given two 2x2 matrices X and Y (as linear combos),
    # returns the list of (coeff_X, coeff_Y) pairs for the 7 Strassen products,
    # along with how they contribute to Z.
    
    # Actually, let's just build the full 49-term decomposition for 4x4*4x4 directly.
    
    # For the 4x4 case, we have 7 "big" products, each being a 2x2*2x2 Strassen.
    # Each big product contributes to specific 2x2 blocks of C.
    
    # Let's index the 4x4 matrices with (block_row, block_col, elem_row, elem_col)
    # where block_row, block_col in {0,1} and elem_row, elem_col in {0,1}.
    # So A[br, bc, er, ec] is the element at row br*2+er, col bc*2+ec.
    
    # For the 4x4 Strassen:
    # Big product k (k=1..7) = X_k * Y_k where X_k, Y_k are 2x2 blocks (linear combos).
    # The result Z_k = X_k * Y_k is a 2x2 matrix.
    # Z_k contributes to C with some coefficient and position.
    
    # C11 = M1 - M5 + M7 + M4  (wait, let me recheck)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1 contributes to C11 (+1) and C22 (+1)
    # M2 contributes to C21 (+1) and C22 (-1)
    # M3 contributes to C12 (+1) and C22 (+1)
    # M4 contributes to C11 (+1) and C21 (+1)
    # M5 contributes to C11 (-1) and C12 (+1)
    # M6 contributes to C22 (+1)
    # M7 contributes to C11 (+1)
    
    # Now, for each M_k = X_k * Y_k, we decompose into 7 Strassen 2x2 products.
    # Each of those 7 products is a scalar product (linear combo of A) * (linear combo of B).
    
    # Let's define the 7 X_k and Y_k as 2x2 matrices of linear combinations.
    
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # M1 = (A11 + A22)(B11 + B22)
    #   X1 = A11 + A22, Y1 = B11 + B22
    # M2 = (A21 + A22)B11
    #   X2 = A21 + A22, Y2 = B11
    # M3 = A11(B12 - B22)
    #   X3 = A11, Y3 = B12 - B22
    # M4 = A22(B21 - B11)
    #   X4 = A22, Y4 = B21 - B11
    # M5 = (A11 + A12)B22
    #   X5 = A11 + A12, Y5 = B22
    # M6 = (A21 - A11)(B11 + B12)
    #   X6 = A21 - A11, Y6 = B11 + B12
    # M7 = (A12 - A22)(B21 + B22)
    #   X7 = A12 - A22, Y7 = B21 + B22
    
    # For each M_k, we do Strassen 2x2 on X_k * Y_k.
    # X_k has elements (x00, x01, x10, x11) which are linear combos of A elements.
    # Y_k has elements (y00, y01, y10, y11) which are linear combos of B elements.
    
    # Let's represent each 2x2 matrix as a dict mapping (er, ec) to a dict of (a_idx, coeff).
    # a_idx is the index in the 4x4 A matrix (0..15), and similarly for B.
    
    # Actually, let's just build the U, V, W matrices directly.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # We'll build the decomposition for the left 4x4 block (B_L) and right 4x4 block (B_R).
    # For B_L, B indices are 0..15 (4x4).
    # For B_R, B indices are 16..31 (4x4), offset by 16.
    
    # Let's define a function that builds the 49 terms for one 4x4*4x4 block.
    
    def build_4x4_terms(b_offset, c_offset):
        """
        Build the 49 terms for 4x4 * 4x4.
        b_offset: offset for B indices (0 for left, 16 for right)
        c_offset: offset for C indices (0 for left, 16 for right)
        Returns list of 49 tuples: (U_indices, V_indices, W_indices)
        where each is a list of (index, coeff) pairs.
        """
        terms = []
        
        # Define the 7 big products
        # Each big product k has X_k and Y_k as 2x2 matrices.
        # X_k elements are linear combos of A (16 elements).
        # Y_k elements are linear combos of B (16 elements, with offset).
        
        # A index: i*4 + j where i,j in 0..3
        # B index: (j*8 + k) + b_offset where j in 0..3, k in 0..3 (for the 4x4 block)
        #   Actually, B is 4x8, so B[j,k] has index j*8 + k.
        #   For the left block (k=0..3), index = j*8 + k.
        #   For the right block (k=4..7), index = j*8 + k = j*8 + 4 + (k-4).
        #   So b_offset for the 4x4 block: for left, k_local = k, index = j*8 + k_local.
        #   For right, k_local = k-4, index = j*8 + 4 + k_local.
        
        # C index: C[i,k] has index k*4 + i (trace convention W[k*n+i,r]).
        #   For left block, k=0..3, index = k*4 + i.
        #   For right block, k=4..7, index = k*4 + i = (4+k_local)*4 + i = 16 + k_local*4 + i.
        
        # Let's define the 7 X_k and Y_k.
        # Each is a 2x2 matrix where each element is a dict: {a_idx: coeff} or {b_idx: coeff}.
        
        # A block (br, bc) where br, bc in {0,1}:
        #   A[br*2+er, bc*2+ec] for er, ec in {0,1}
        #   Index in A: (br*2+er)*4 + (bc*2+ec)
        
        # Let's define helper to get A element index.
        def a_idx(br, bc, er, ec):
            return (br*2 + er)*4 + (bc*2 + ec)
        
        def b_idx(bc, er, ec):
            # B block (bc, er, ec) within the 4x4 block
            # B row = er, B col = bc*2 + ec (within the 4x4 block)
            # Global B index = row*8 + col + b_offset_col
            # Wait, B is 4x8. B[j,k]. j in 0..3, k in 0..7.
            # For the 4x4 block, j = er (0..3), k = bc*2+ec + b_col_offset
            # b_col_offset = 0 for left, 4 for right.
            row = er
            col = bc*2 + ec
            return row*8 + col + b_offset
            
        def c_idx(bc, br, er, ec):
            # C block (bc, br) where bc is col block, br is row block
            # C[i,k] where i = br*2+er, k = bc*2+ec + c_col_offset
            # c_col_offset = 0 for left, 4 for right.
            # Index = k*4 + i = (bc*2+ec + c_col_offset)*4 + (br*2+er)
            row = br*2 + er
            col = bc*2 + ec + c_offset
            return col*4 + row
        
        # Now define the 7 big products.
        # Each X_k and Y_k is a 2x2 matrix of dicts.
        
        # M1: X1 = A11 + A22, Y1 = B11 + B22
        # A11 = block (0,0), A22 = block (1,1)
        # B11 = block (0,0), B22 = block (1,1)
        
        big_products = []
        
        # M1
        X1 = {}
        for er in range(2):
            for ec in range(2):
                idx_x = (er, ec)
                # A11[er,ec] + A22[er,ec]
                a11 = a_idx(0, 0, er, ec)
                a22 = a_idx(1, 1, er, ec)
                X1[idx_x] = {a11: 1, a22: 1}
        Y1 = {}
        for er in range(2):
            for ec in range(2):
                idx_y = (er, ec)
                b11 = b_idx(0, er, ec)
                b22 = b_idx(1, er, ec)
                Y1[idx_y] = {b11: 1, b22: 1}
        big_products.append((X1, Y1))
        
        # M2: X2 = A21 + A22, Y2 = B11
        X2 = {}
        for er in range(2):
            for ec in range(2):
                a21 = a_idx(1, 0, er, ec)
                a22 = a_idx(1, 1, er, ec)
                X2[(er, ec)] = {a21: 1, a22: 1}
        Y2 = {}
        for er in range(2):
            for ec in range(2):
                b11 = b_idx(0, er, ec)
                Y2[(er, ec)] = {b11: 1}
        big_products.append((X2, Y2))
        
        # M3: X3 = A11, Y3 = B12 - B22
        X3 = {}
        for er in range(2):
            for ec in range(2):
                a11 = a_idx(0, 0, er, ec)
                X3[(er, ec)] = {a11: 1}
        Y3 = {}
        for er in range(2):
            for ec in range(2):
                b12 = b_idx(1, er, ec)
                b22 = b_idx(1, er, ec)  # Wait, B12 is block (0,1), B22 is block (1,1)
                # B12[er,ec] - B22[er,ec]
                b12_idx = b_idx(1, er, ec)  # col block 1
                b22_idx = b_idx(1, er, ec)  # row block 1, col block 1
                # Oops, B12 is (row block 0, col block 1), B22 is (row block 1, col block 1)
                # Let me redefine b_idx to take (row_block, col_block, er, ec)
                pass
        
        # Let me redefine b_idx and c_idx more carefully.
        
        # B is 4x8. B[j,k]. j in 0..3, k in 0..7.
        # For a 4x4 block, we have j in 0..3, k in 0..3 (left) or 4..7 (right).
        # Within the 4x4 block, we divide into 2x2 blocks.
        # Block (rb, cb) where rb, cb in {0,1}:
        #   j = rb*2 + er, k = cb*2 + ec + k_offset
        #   where k_offset = 0 for left, 4 for right.
        #   B index = j*8 + k = (rb*2+er)*8 + (cb*2+ec+k_offset)
        
        def b_idx2(rb, cb, er, ec):
            j = rb*2 + er
            k = cb*2 + ec + b_offset
            return j*8 + k
        
        def c_idx2(rb, cb, er, ec):
            i = rb*2 + er
            k = cb*2 + ec + c_offset
            return k*4 + i
        
        # Now redo the big products.
        
        big_products = []
        
        # M1: X1 = A11 + A22, Y1 = B11 + B22
        X1 = {}
        for er in range(2):
            for ec in range(2):
                a11 = a_idx(0, 0, er, ec)
                a22 = a_idx(1, 1, er, ec)
                X1[(er, ec)] = {a11: 1, a22: 1}
        Y1 = {}
        for er in range(2):
            for ec in range(2):
                b11 = b_idx2(0, 0, er, ec)
                b22 = b_idx2(1, 1, er, ec)
                Y1[(er, ec)] = {b11: 1, b22: 1}
        big_products.append((X1, Y1))
        
        # M2: X2 = A21 + A22, Y2 = B11
        X2 = {}
        for er in range(2):
            for ec in range(2):
                a21 = a_idx(1, 0, er, ec)
                a22 = a_idx(1, 1, er, ec)
                X2[(er, ec)] = {a21: 1, a22: 1}
        Y2 = {}
        for er in range(2):
            for ec in range(2):
                b11 = b_idx2(0, 0, er, ec)
                Y2[(er, ec)] = {b11: 1}
        big_products.append((X2, Y2))
        
        # M3: X3 = A11, Y3 = B12 - B22
        X3 = {}
        for er in range(2):
            for ec in range(2):
                a11 = a_idx(0, 0, er, ec)
                X3[(er, ec)] = {a11: 1}
        Y3 = {}
        for er in range(2):
            for ec in range(2):
                b12 = b_idx2(0, 1, er, ec)
                b22 = b_idx2(1, 1, er, ec)
                Y3[(er, ec)] = {b12: 1, b22: -1}
        big_products.append((X3, Y3))
        
        # M4: X4 = A22, Y4 = B21 - B11
        X4 = {}
        for er in range(2):
            for ec in range(2):
                a22 = a_idx(1, 1, er, ec)
                X4[(er, ec)] = {a22: 1}
        Y4 = {}
        for er in range(2):
            for ec in range(2):
                b21 = b_idx2(1, 0, er, ec)
                b11 = b_idx2(0, 0, er, ec)
                Y4[(er, ec)] = {b21: 1, b11: -1}
        big_products.append((X4, Y4))
        
        # M5: X5 = A11 + A12, Y5 = B22
        X5 = {}
        for er in range(2):
            for ec in range(2):
                a11 = a_idx(0, 0, er, ec)
                a12 = a_idx(0, 1, er, ec)
                X5[(er, ec)] = {a11: 1, a12: 1}
        Y5 = {}
        for er in range(2):
            for ec in range(2):
                b22 = b_idx2(1, 1, er, ec)
                Y5[(er, ec)] = {b22: 1}
        big_products.append((X5, Y5))
        
        # M6: X6 = A21 - A11, Y6 = B11 + B12
        X6 = {}
        for er in range(2):
            for ec in range(2):
                a21 = a_idx(1, 0, er, ec)
                a11 = a_idx(0, 0, er, ec)
                X6[(er, ec)] = {a21: 1, a11: -1}
        Y6 = {}
        for er in range(2):
            for ec in range(2):
                b11 = b_idx2(0, 0, er, ec)
                b12 = b_idx2(0, 1, er, ec)
                Y6[(er, ec)] = {b11: 1, b12: 1}
        big_products.append((X6, Y6))
        
        # M7: X7 = A12 - A22, Y7 = B21 + B22
        X7 = {}
        for er in range(2):
            for ec in range(2):
                a12 = a_idx(0, 1, er, ec)
                a22 = a_idx(1, 1, er, ec)
                X7[(er, ec)] = {a12: 1, a22: -1}
        Y7 = {}
        for er in range(2):
            for ec in range(2):
                b21 = b_idx2(1, 0, er, ec)
                b22 = b_idx2(1, 1, er, ec)
                Y7[(er, ec)] = {b21: 1, b22: 1}
        big_products.append((X7, Y7))
        
        # Now, for each big product k, decompose X_k * Y_k using Strassen 2x2.
        # The 7 Strassen products for X*Y:
        # N1 = (x00+x11)(y00+y11)
        # N2 = (x10+x11)y00
        # N3 = x00(y01-y11)
        # N4 = x11(y10-y00)
        # N5 = (x00+x01)y11
        # N6 = (x10-x00)(y00+y01)
        # N7 = (x01-x11)(y10+y11)
        
        # Each N_i is a product of two linear combinations.
        # We need to express each as sum of scalar products.
        
        # Let's define a helper that takes two dicts (for x and y elements)
        # and returns the list of (a_indices, b_indices) for the product.
        
        def product_terms(x_dict, y_dict):
            """
            x_dict: dict mapping a_idx to coeff
            y_dict: dict mapping b_idx to coeff
            Returns list of (a_idx, b_idx, coeff) for all scalar products.
            """
            terms = []
            for a_idx, ca in x_dict.items():
                for b_idx, cb in y_dict.items():
                    terms.append((a_idx, b_idx, ca * cb))
            return terms
        
        # For each big product, we have X and Y as 2x2 dicts.
        # We compute the 7 Strassen products.
        
        for big_k, (X, Y) in enumerate(big_products):
            # X and Y are dicts: (er, ec) -> {idx: coeff}
            
            # N1 = (x00+x11)(y00+y11)
            x_n1 = {}
            for idx, coeff in X[(0,0)].items():
                x_n1[idx] = x_n1.get(idx, 0) + coeff
            for idx, coeff in X[(1,1)].items():
                x_n1[idx] = x_n1.get(idx, 0) + coeff
            y_n1 = {}
            for idx, coeff in Y[(0,0)].items():
                y_n1[idx] = y_n1.get(idx, 0) + coeff
            for idx, coeff in Y[(1,1)].items():
                y_n1[idx] = y_n1.get(idx, 0) + coeff
            terms_n1 = product_terms(x_n1, y_n1)
            
            # N2 = (x10+x11)y00
            x_n2 = {}
            for idx, coeff in X[(1,0)].items():
                x_n2[idx] = x_n2.get(idx, 0) + coeff
            for idx, coeff in X[(1,1)].items():
                x_n2[idx] = x_n2.get(idx, 0) + coeff
            y_n2 = Y[(0,0)]
            terms_n2 = product_terms(x_n2, y_n2)
            
            # N3 = x00(y01-y11)
            x_n3 = X[(0,0)]
            y_n3 = {}
            for idx, coeff in Y[(0,1)].items():
                y_n3[idx] = y_n3.get(idx, 0) + coeff
            for idx, coeff in Y[(1,1)].items():
                y_n3[idx] = y_n3.get(idx, 0) - coeff
            terms_n3 = product_terms(x_n3, y_n3)
            
            # N4 = x11(y10-y00)
            x_n4 = X[(1,1)]
            y_n4 = {}
            for idx, coeff in Y[(1,0)].items():
                y_n4[idx] = y_n4.get(idx, 0) + coeff
            for idx, coeff in Y[(0,0)].items():
                y_n4[idx] = y_n4.get(idx, 0) - coeff
            terms_n4 = product_terms(x_n4, y_n4)
            
            # N5 = (x00+x01)y11
            x_n5 = {}
            for idx, coeff in X[(0,0)].items():
                x_n5[idx] = x_n5.get(idx, 0) + coeff
            for idx, coeff in X[(0,1)].items():
                x_n5[idx] = x_n5.get(idx, 0) + coeff
            y_n5 = Y[(1,1)]
            terms_n5 = product_terms(x_n5, y_n5)
            
            # N6 = (x10-x00)(y00+y01)
            x_n6 = {}
            for idx, coeff in X[(1,0)].items():
                x_n6[idx] = x_n6.get(idx, 0) + coeff
            for idx, coeff in X[(0,0)].items():
                x_n6[idx] = x_n6.get(idx, 0) - coeff
            y_n6 = {}
            for idx, coeff in Y[(0,0)].items():
                y_n6[idx] = y_n6.get(idx, 0) + coeff
            for idx, coeff in Y[(0,1)].items():
                y_n6[idx] = y_n6.get(idx, 0) + coeff
            terms_n6 = product_terms(x_n6, y_n6)
            
            # N7 = (x01-x11)(y10+y11)
            x_n7 = {}
            for idx, coeff in X[(0,1)].items():
                x_n7[idx] = x_n7.get(idx, 0) + coeff
            for idx, coeff in X[(1,1)].items():
                x_n7[idx] = x_n7.get(idx, 0) - coeff
            y_n7 = {}
            for idx, coeff in Y[(1,0)].items():
                y_n7[idx] = y_n7.get(idx, 0) + coeff
            for idx, coeff in Y[(1,1)].items():
                y_n7[idx] = y_n7.get(idx, 0) + coeff
            terms_n7 = product_terms(x_n7, y_n7)
            
            # Now, the Strassen 2x2 result Z = X*Y has:
            # z00 = N1 + N4 - N5 + N7
            # z01 = N3 + N5
            # z10 = N2 + N4
            # z11 = N1 - N2 + N3 + N6
            
            # The big product M_{big_k} = Z, which is a 2x2 matrix.
            # Z contributes to C blocks with certain coefficients.
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for big_k (0-indexed: 0=M1, 1=M2, ..., 6=M7):
            # We need to determine the coefficient and target C block.
            
            # Let's define a dict: big_k -> list of (c_block_row, c_block_col, coeff)
            # c_block_row, c_block_col in {0,1}
            
            contrib = {
                0: [(0,0,1), (1,1,1)],       # M1 -> C11(+1), C22(+1)
                1: [(1,0,1), (1,1,-1)],      # M2 -> C21(+1), C22(-1)
                2: [(0,1,1), (1,1,1)],       # M3 -> C12(+1), C22(+1)
                3: [(0,0,1), (1,0,1)],       # M4 -> C11(+1), C21(+1)
                4: [(0,0,-1), (0,1,1)],      # M5 -> C11(-1), C12(+1)
                5: [(1,1,1)],                # M6 -> C22(+1)
                6: [(0,0,1)],                # M7 -> C11(+1)
            }
            
            # For each of the 7 Strassen products (N1..N7), and for each contribution
            # of the big product to C, we add terms to U, V, W.
            
            # The 7 Strassen products contribute to Z as:
            # z00 = N1 + N4 - N5 + N7
            # z01 = N3 + N5
            # z10 = N2 + N4
            # z11 = N1 - N2 + N3 + N6
            
            # Z is a 2x2 matrix. Z[er, ec] for er, ec in {0,1}.
            # Z[er, ec] contributes to C block (c_br, c_bc) at position (er, ec).
            # C index = c_idx2(c_br, c_bc, er, ec)
            
            # For each Strassen product i (0..6 for N1..N7), we have terms_i.
            # For each (a_idx, b_idx, coeff) in terms_i, and for each contribution
            # of the big product to C, and for each (er, ec) in Z where N_i appears
            # with coefficient c_ze, we add:
            #   U[a_idx] += coeff * c_contrib * c_ze
            #   V[b_idx] += coeff * c_contrib * c_ze
            #   W[c_idx] += coeff * c_contrib * c_ze
            
            # Let's define the contribution of each N_i to each Z[er, ec].
            # N1: z00(+1), z11(+1)
            # N2: z10(+1), z11(-1)
            # N3: z01(+1), z11(+1)
            # N4: z00(+1), z10(+1)
            # N5: z00(-1), z01(+1)
            # N6: z11(+1)
            # N7: z00(+1)
            
            n_contrib = {
                0: [(0,0,1), (1,1,1)],       # N1
                1: [(1,0,1), (1,1,-1)],      # N2
                2: [(0,1,1), (1,1,1)],       # N3
                3: [(0,0,1), (1,0,1)],       # N4
                4: [(0,0,-1), (0,1,1)],      # N5
                5: [(1,1,1)],                # N6
                6: [(0,0,1)],                # N7
            }
            
            all_terms = [terms_n1, terms_n2, terms_n3, terms_n4, terms_n5, terms_n6, terms_n7]
            
            for n_i in range(7):
                for (a_idx, b_idx, coeff) in all_terms[n_i]:
                    for (c_br, c_bc, c_coeff) in contrib[big_k]:
                        for (z_er, z_ec, z_coeff) in n_contrib[n_i]:
                            c_idx = c_idx2(c_br, c_bc, z_er, z_ec)
                            total_coeff = coeff * c_coeff * z_coeff
                            terms.append((a_idx, b_idx, c_idx, total_coeff))
        
        return terms
    
    # Build terms for left block (b_offset=0, c_offset=0) and right block (b_offset=4, c_offset=4)
    all_terms = []
    all_terms.extend(build_4x4_terms(0, 0))
    all_terms.extend(build_4x4_terms(4, 4))
    
    # Now, fill U, V, W.
    # Each term is (a_idx, b_idx, c_idx, coeff).
    # We assign each term to a unique column in U, V, W.
    
    for col, (a_idx, b_idx, c_idx, coeff) in enumerate(all_terms):
        U[a_idx][col] = coeff
        V[b_idx][col] = coeff
        W[c_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

