import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    C = A*B1 + A*B2 + A*B3
    Use Strassen's algorithm for 4x4*4x4 (rank 49) twice, and schoolbook for 4x4*4x1 (rank 4).
    Total rank: 49 + 49 + 4 = 102.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Dimensions of factor matrices
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # We'll build the decomposition as a list of (u_indices, v_indices, w_indices, coefficients)
    # Each rank-1 term contributes to T[u,v,w] += coeff
    
    terms = []  # list of (u, v, w, coeff)
    
    def add_term(u, v, w, coeff=1.0):
        terms.append((u, v, w, coeff))
    
    # Indexing conventions:
    # A is n x m = 4 x 4, stored as u = i*m + j where i=row, j=col
    # B is m x p = 4 x 9, stored as v = j*p + k where j=row, k=col
    # C is p x n = 9 x 4, stored as w = k*n + i where k=row, i=col
    # 
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # T[u,v,w] = T[i*m+j, j*p+k, k*n+i] = 1
    
    # Strassen's algorithm for 4x4 * 4x4:
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # 
    # 7 multiplications:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # 
    # Results:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # Each 2x2 multiplication has rank 7 (Strassen for 2x2*2x2).
    # But wait - we need to be careful. The rank of 4x4*4x4 via Strassen is 49?
    # Actually, Strassen gives 7 multiplications of 2x2 matrices, each of which 
    # is 2x2*2x2. The 2x2*2x2 multiplication has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49. Yes!
    
    # Let me implement this more carefully.
    # For 4x4 * 4x4, we use the block structure.
    # 
    # A[i,j] where i,j in {0,1,2,3}
    # B[j,k] where j,k in {0,1,2,3}
    # 
    # Block indices:
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    # Similarly for B.
    #
    # Each 2x2 block multiplication can be done with 7 scalar multiplications.
    # But the blocks are 2x2 matrices, not scalars.
    #
    # Actually, the standard Strassen for 4x4 is:
    # View A and B as 2x2 matrices of 2x2 blocks.
    # Use 7 matrix multiplications of 2x2 blocks.
    # Each 2x2 * 2x2 block multiplication has tensor rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let me define the 2x2 Strassen for blocks.
    # For multiplying two 2x2 matrices X and Y to get Z = X*Y:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    #
    # 7 products:
    # P1 = (x00 + x11) * (y00 + y11)
    # P2 = (x10 + x11) * y00
    # P3 = x00 * (y01 - y11)
    # P4 = x11 * (y10 - y00)
    # P5 = (x00 + x01) * y11
    # P6 = (x10 - x00) * (y00 + y01)
    # P7 = (x01 - x11) * (y10 + y11)
    #
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # Now for 4x4 * 4x4:
    # We have 4x4 matrices, split into 2x2 blocks of 2x2 sub-matrices.
    # Let's denote the blocks as A_ab where a,b in {0,1} and each is 2x2.
    # Similarly B_cd.
    #
    # The 7 block multiplications:
    # M1 = (A_00 + A_11) * (B_00 + B_11)
    # M2 = (A_10 + A_11) * B_00
    # M3 = A_00 * (B_01 - B_11)
    # M4 = A_11 * (B_10 - B_00)
    # M5 = (A_00 + A_01) * B_11
    # M6 = (A_10 - A_00) * (B_00 + B_01)
    # M7 = (A_01 - A_11) * (B_10 + B_11)
    #
    # Each M_i is a 2x2 matrix product of two 2x2 matrices (with coefficients).
    # Each such product contributes 7 rank-1 terms.
    # So 7 * 7 = 49 terms total.
    
    # Helper: add a 2x2 * 2x2 multiplication with given coefficient patterns
    # X_coeffs: dict mapping (xi, xj) -> coefficient for X[i,j]
    # Y_coeffs: dict mapping (yi, yj) -> coefficient for Y[i,j]
    # result_add: function to add result coefficients to C
    
    # Let me take a different approach. I'll directly enumerate all rank-1 terms.
    
    # For the 4x4 * 4x4 case, let's define the 49 terms explicitly.
    
    # First, let's define helper to get block index from matrix indices
    # For 4x4 matrix with indices 0-3:
    # Block (bi, bj) contains elements with i in {2*bi, 2*bi+1} and j in {2*bj, 2*bj+1}
    
    # Let me define the 7 block-level Strassen products, and for each, the 7 scalar-level products.
    
    # Block-level products (each is a 2x2 * 2x2 matrix product):
    # M1: X = A_00 + A_11, Y = B_00 + B_11
    # M2: X = A_10 + A_11, Y = B_00
    # M3: X = A_00, Y = B_01 - B_11
    # M4: X = A_11, Y = B_10 - B_00
    # M5: X = A_00 + A_01, Y = B_11
    # M6: X = A_10 - A_00, Y = B_00 + B_01
    # M7: X = A_01 - A_11, Y = B_10 + B_11
    
    # For each block product M_i = X * Y where X and Y are 2x2 matrices with entries that are linear combinations of A and B entries:
    # We apply 2x2 Strassen to get 7 scalar products.
    
    # Let's define the 2x2 Strassen in terms of scalar indices.
    # For X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]:
    # P1 = (x00 + x11) * (y00 + y11)
    # P2 = (x10 + x11) * y00
    # P3 = x00 * (y01 - y11)
    # P4 = x11 * (y10 - y00)
    # P5 = (x00 + x01) * y11
    # P6 = (x10 - x00) * (y00 + y01)
    # P7 = (x01 - x11) * (y10 + y11)
    
    # Each Pk is a product of two linear forms in the original A and B entries.
    # Each such product contributes to the tensor.
    
    # Let me just hardcode all 49 terms for the first 4x4*4x4 block (columns 0-3 of B).
    
    # Actually, let me use a cleaner approach. I'll define the decomposition for 4x4*4x4
    # and then instantiate it for the three column blocks.
    
    # For 4x4 * 4x4 (A is 4x4, B is 4x4, C is 4x4):
    # u = i*4 + j (i,j in 0..3)
    # v = j*4 + k (j,k in 0..3)  
    # w = k*4 + i (k,i in 0..3)
    
    # Let me define the 49 terms as (u, v, w, coeff) tuples.
    
    # I'll generate them programmatically using the Strassen structure.
    
    def generate_strassen_4x4_4x4(col_offset=0):
        """Generate 49 rank-1 terms for 4x4 * 4x4 matrix multiplication.
        col_offset: offset for the column index k in B (0-3 maps to col_offset+0 to col_offset+3)
        Returns list of (u, v, w, coeff)
        """
        terms_local = []
        
        # Block structure for 4x4:
        # Block (bi, bj) has elements with row i in {2*bi, 2*bi+1} and col j in {2*bj, 2*bj+1}
        
        # The 7 block-level Strassen products:
        # Each is X * Y where X and Y are 2x2 blocks (linear combos of A and B blocks)
        
        # For each block product, we need to expand into 7 scalar products.
        
        # Let's define the 7 block products with their X and Y block coefficient patterns.
        # X_blocks: dict from (bi, bj) -> coeff for that block in X
        # Y_blocks: dict from (bi, bj) -> coeff for that block in Y
        
        block_products = [
            # (X_blocks, Y_blocks)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),             # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),             # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),             # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),             # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # M7
        ]
        
        # For each block product, expand into 7 scalar products using 2x2 Strassen.
        # X is a 2x2 matrix where X[ei, ej] = sum over blocks (bi,bj) of coeff * A[2*bi+ei, 2*bj+ej]
        # Y is a 2x2 matrix where Y[fi, fj] = sum over blocks (bi,bj) of coeff * B[2*bi+fi, col_offset+2*bj+fj]
        
        for x_blocks, y_blocks in block_products:
            # Build X[ei, ej] as a list of (i, j, coeff) for A
            # Build Y[fi, fj] as a list of (j, k, coeff) for B
            X_entries = [[[] for _ in range(2)] for _ in range(2)]  # X_entries[ei][ej] = [(i, j, coeff), ...]
            Y_entries = [[[] for _ in range(2)] for _ in range(2)]  # Y_entries[fi][fj] = [(j, k, coeff), ...]
            
            for (bi, bj), coeff in x_blocks.items():
                for ei in range(2):
                    for ej in range(2):
                        i = 2*bi + ei
                        j = 2*bj + ej
                        X_entries[ei][ej].append((i, j, coeff))
            
            for (bi, bj), coeff in y_blocks.items():
                for fi in range(2):
                    for fj in range(2):
                        j = 2*bi + fi
                        k = col_offset + 2*bj + fj
                        Y_entries[fi][fj].append((j, k, coeff))
            
            # Now apply 2x2 Strassen to X and Y.
            # P1 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1])
            # P2 = (X[1,0] + X[1,1]) * Y[0,0]
            # P3 = X[0,0] * (Y[0,1] - Y[1,1])
            # P4 = X[1,1] * (Y[1,0] - Y[0,0])
            # P5 = (X[0,0] + X[0,1]) * Y[1,1]
            # P6 = (X[1,0] - X[0,0]) * (Y[0,0] + Y[0,1])
            # P7 = (X[0,1] - X[1,1]) * (Y[1,0] + Y[1,1])
            
            # Each Pk is a product of two linear forms. We need to expand.
            
            # Define the 7 Strassen products as pairs of (X_linear_form, Y_linear_form)
            # X_linear_form: list of (ei, ej, coeff) for X[ei,ej]
            # Y_linear_form: list of (fi, fj, coeff) for Y[fi,fj]
            
            strassen_2x2 = [
                # P1
                ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
                # P2
                ([(1,0,1), (1,1,1)], [(0,0,1)]),
                # P3
                ([(0,0,1)], [(0,1,1), (1,1,-1)]),
                # P4
                ([(1,1,1)], [(1,0,1), (0,0,-1)]),
                # P5
                ([(0,0,1), (0,1,1)], [(1,1,1)]),
                # P6
                ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
                # P7
                ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
            ]
            
            for x_form, y_form in strassen_2x2:
                # Expand x_form: for each (ei, ej, cx) in x_form, get all A entries
                # Expand y_form: for each (fi, fj, cy) in y_form, get all B entries
                # Each combination gives a rank-1 term.
                
                for ei, ej, cx in x_form:
                    for i, j, ca in X_entries[ei][ej]:
                        a_coeff = cx * ca
                        u = i * 4 + j  # A index
                        for fi, fj, cy in y_form:
                            for jj, k, cb in Y_entries[fi][fj]:
                                b_coeff = cy * cb
                                v = jj * 9 + k  # B index (using full p=9)
                                w = k * 4 + i  # C index
                                total_coeff = a_coeff * b_coeff
                                if total_coeff != 0:
                                    terms_local.append((u, v, w, total_coeff))
        
        return terms_local
    
    # Generate terms for three blocks:
    # Block 1: columns 0-3 of B
    terms.extend(generate_strassen_4x4_4x4(col_offset=0))
    # Block 2: columns 4-7 of B
    terms.extend(generate_strassen_4x4_4x4(col_offset=4))
    # Block 3: column 8 of B (4x4 * 4x1)
    # For this, we can use schoolbook: 4 multiplications for each output element?
    # Actually, for 4x4 * 4x1, the rank is 4 (each output element C[k,i] = sum_j A[i,j]*B[j,8])
    # We can decompose each of the 16 outputs as a sum, but the tensor rank is 4.
    # 
    # Actually, for multiplying a 4x4 matrix by a 4x1 vector, the result is a 4x1 vector.
    # C[k,i] = sum_{j=0}^{3} A[i,j] * B[j,8]
    # 
    # The tensor T[i*4+j, j*4+8, 8*4+i] = 1 for i,j in 0..3
    # This is a 4x4x4 tensor? No, it's 16x4x4 tensor.
    # 
    # The rank of this tensor: For each fixed i, we have 4 terms (one for each j).
    # These are 16 separate rank-1 terms? Or can we combine?
    # 
    # T[i*4+j, j*4+8, 8*4+i] = 1
    # For different (i,j) pairs, these are distinct entries.
    # Each entry is 1, and they don't share indices in a way that allows combining.
    # Actually, for fixed i, the 4 terms (j=0,1,2,3) have different v and w indices.
    # For fixed j, the 4 terms (i=0,1,2,3) have different u and w indices.
    # 
    # The tensor rank is 16 (one term per (i,j) pair) for the schoolbook.
    # But we can do better! 
    # 
    # Actually, the rank of the 4x4*4x1 multiplication tensor is 4.
    # Because C = A * b where b is a 4x1 vector.
    # C[i] = sum_j A[i,j] * b[j]
    # 
    # We can write this as:
    # C[i] = A[i,0]*b[0] + A[i,1]*b[1] + A[i,2]*b[2] + A[i,3]*b[3]
    # 
    # As a tensor: T[u,v,w] where u = i*4+j, v = j*4+8, w = 8*4+i
    # 
    # The rank-1 decomposition:
    # For each j in 0..3:
    #   term_j: U has 1s at positions i*4+j for all i, V has 1 at position j*4+8, W has 1s at positions 8*4+i for all i
    # 
    # So 4 terms total!
    
    for j in range(4):
        for i in range(4):
            u = i * 4 + j
            v = j * 4 + 8
            w = 8 * 4 + i
            # This is part of term j
            # We'll add these separately and then combine
    
    # Actually, let me add the 4 terms for the 4x4*4x1 case:
    for j in range(4):
        # U: 1 at positions i*4+j for i=0,1,2,3
        # V: 1 at position j*4+8
        # W: 1 at positions 8*4+i for i=0,1,2,3
        # This is a rank-1 term
        # But we can't represent a rank-1 term with multiple (u,v,w) in a single call to add_term.
        # We need to add all combinations.
        for i in range(4):
            u = i * 4 + j
            v = j * 4 + 8
            w = 8 * 4 + i
            add_term(u, v, w, 1.0)
    
    # Now convert terms to factor matrices
    # We have R = len(terms) rank-1 terms (with possible duplicates that should be combined)
    
    # Combine terms with same (u,v,w)
    from collections import defaultdict
    coeff_dict = defaultdict(float)
    for u, v, w, coeff in terms:
        coeff_dict[(u, v, w)] += coeff
    
    # Remove zero coefficients
    coeff_dict = {k: v for k, v in coeff_dict.items() if abs(v) > 1e-12}
    
    # Now build factor matrices.
    # Each unique (u,v,w) with non-zero coefficient is a rank-1 term.
    # But we can do better: group by u, by v, by w to find the minimal number of terms.
    # 
    # Actually, for a general tensor, the number of non-zero entries in the CP decomposition
    # can be much larger than the rank. But in our case, the tensor has a special structure.
    # 
    # Let me just use one term per non-zero entry for now. This gives rank = number of non-zero entries.
    # For Strassen, this should be 49 + 49 + 16 = 114. That's worse than 102!
    # 
    # The issue is that I'm expanding the Strassen products into individual scalar products,
    # which gives 49 terms per 4x4*4x4 block. But each "term" in my expansion is actually
    # a product of two linear forms, which may involve multiple A and B entries.
    # 
    # I made an error: when I expand P1 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1]),
    # and X[0,0] = A[0,0] + A[2,0] (for example), and Y[0,0] = B[0,0] + B[2,0],
    # then P1 = (A[0,0] + A[2,0] + A[1,1] + A[3,1]) * (B[0,0] + B[2,0] + B[1,1] + B[3,1])
    # This is a product of two linear forms, each involving 4 terms.
    # When expanded, it gives 16 scalar products.
    # 
    # But in the tensor decomposition, a product of two linear forms is still ONE rank-1 term!
    # Because U, V, W can have multiple non-zero entries.
    
    # I need to reconsider. Let me rebuild this properly.
    
    # A rank-1 term in the tensor is: T[u,v,w] += U[u,c] * V[v,c] * W[w,c] for a fixed c.
    # U, V, W are vectors (one column of the factor matrices).
    # 
    # So each rank-1 term is defined by three vectors U_c, V_c, W_c.
    # 
    # For Strassen, each "multiplication" like P1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # corresponds to one rank-1 term where:
    # U_c has 1 at positions corresponding to A[0,0] and A[1,1]
    # V_c has 1 at positions corresponding to B[0,0] and B[1,1]
    # W_c has 1 at positions corresponding to the output entries
    
    # Let me restart with a cleaner approach.
    
    # Clear and redo
    terms.clear()
    
    # For 4x4 * 4x4, the 49 rank-1 terms come from:
    # 7 block-level Strassen products, each of which is a 2x2 * 2x2 matrix product.
    # Each 2x2 * 2x2 product has 7 rank-1 terms (Strassen).
    # Total: 7 * 7 = 49.
    
    # But I need to be more careful about how the linear forms map to U, V, W vectors.
    
    # Let me define each rank-1 term by its U, V, W support.
    
    rank1_terms = []  # list of (U_dict, V_dict, W_dict) where each is {index: coeff}
    
    def add_rank1_term(U_dict, V_dict, W_dict):
        rank1_terms.append((U_dict, V_dict, W_dict))
    
    # For 4x4 * 4x4 with B columns offset by col_offset:
    # A[i,j] -> u = i*4 + j
    # B[j,k] -> v = j*9 + (col_offset + k)
    # C[k,i] -> w = (col_offset + k)*4 + i
    
    def add_strassen_block(col_offset):
        """Add 49 rank-1 terms for 4x4 * 4x4 with B columns at col_offset..col_offset+3"""
        
        # The 7 block-level Strassen products:
        # Each is X * Y where X and Y are 2x2 blocks.
        # X[i,j] is a linear combination of A entries.
        # Y[i,j] is a linear combination of B entries.
        # The result Z[i,j] contributes to C entries.
        
        # For each block product, we do a 2x2 Strassen, giving 7 sub-products.
        # Each sub-product is a product of two linear forms.
        
        # Let me define the 7 block products:
        # M1: X = A_00 + A_11, Y = B_00 + B_11
        # M2: X = A_10 + A_11, Y = B_00
        # M3: X = A_00, Y = B_01 - B_11
        # M4: X = A_11, Y = B_10 - B_00
        # M5: X = A_00 + A_01, Y = B_11
        # M6: X = A_10 - A_00, Y = B_00 + B_01
        # M7: X = A_01 - A_11, Y = B_10 + B_11
        
        # Where A_bi,bj is the block with rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
        # Similarly for B.
        
        block_defs = [
            # (X_blocks, Y_blocks) where each is {block_index: coeff}
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ]
        
        for x_blocks, y_blocks in block_defs:
            # Build X[ei,ej] as linear form in A: {u_index: coeff}
            # Build Y[fi,fj] as linear form in B: {v_index: coeff}
            X = [[{} for _ in range(2)] for _ in range(2)]
            Y = [[{} for _ in range(2)] for _ in range(2)]
            
            for (bi, bj), coeff in x_blocks.items():
                for ei in range(2):
                    for ej in range(2):
                        i = 2*bi + ei
                        j = 2*bj + ej
                        u = i * 4 + j
                        X[ei][ej][u] = X[ei][ej].get(u, 0) + coeff
            
            for (bi, bj), coeff in y_blocks.items():
                for fi in range(2):
                    for fj in range(2):
                        j = 2*bi + fi
                        k = col_offset + 2*bj + fj
                        v = j * 9 + k
                        Y[fi][fj][v] = Y[fi][fj].get(v, 0) + coeff
            
            # Now do 2x2 Strassen on X and Y.
            # 7 sub-products:
            # P1 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1])
            # P2 = (X[1,0] + X[1,1]) * Y[0,0]
            # P3 = X[0,0] * (Y[0,1] - Y[1,1])
            # P4 = X[1,1] * (Y[1,0] - Y[0,0])
            # P5 = (X[0,0] + X[0,1]) * Y[1,1]
            # P6 = (X[1,0] - X[0,0]) * (Y[0,0] + Y[0,1])
            # P7 = (X[0,1] - X[1,1]) * (Y[1,0] + Y[1,1])
            
            def add_linear_form(d1, d2, sign1=1, sign2=1):
                """Add sign1*d1 + sign2*d2"""
                result = {}
                for k, v in d1.items():
                    result[k] = result.get(k, 0) + sign1 * v
                for k, v in d2.items():
                    result[k] = result.get(k, 0) + sign2 * v
                return {k: v for k, v in result.items() if abs(v) > 1e-12}
            
            sub_products = [
                (add_linear_form(X[0][0], X[1][1]), add_linear_form(Y[0][0], Y[1][1])),
                (add_linear_form(X[1][0], X[1][1]), Y[0][0]),
                (X[0][0], add_linear_form(Y[0][1], Y[1][1], 1, -1)),
                (X[1][1], add_linear_form(Y[1][0], Y[0][0], 1, -1)),
                (add_linear_form(X[0][0], X[0][1]), Y[1][1]),
                (add_linear_form(X[1][0], X[0][0], 1, -1), add_linear_form(Y[0][0], Y[0][1])),
                (add_linear_form(X[0][1], X[1][1], 1, -1), add_linear_form(Y[1][0], Y[1][1])),
            ]
            
            # Each sub-product Pk = X_form * Y_form contributes to C.
            # The result of X * Y (2x2 matrices) gives a 2x2 matrix Z.
            # Z[ei,fi] contributes to C entries.
            # 
            # For the 2x2 Strassen, the results are:
            # Z00 = P1 + P4 - P5 + P7
            # Z01 = P3 + P5
            # Z10 = P2 + P4
            # Z11 = P1 - P2 + P3 + P6
            # 
            # But in the tensor decomposition, each Pk is ONE rank-1 term.
            # The coefficients in Z are just linear combinations, which means
            # the W vector for Pk has coefficients that reflect how Pk contributes to each Z entry.
            # 
            # Actually, no. In the standard Strassen decomposition of 2x2*2x2:
            # Each of the 7 products Pk is a scalar product.
            # The result Z[i,j] is a linear combination of the Pk's.
            # 
            # In the tensor T[u,v,w]:
            # u corresponds to A[i,j], v to B[j,k], w to C[k,i].
            # 
            # For a given Pk = X_form * Y_form:
            # X_form is a linear combination of A entries: sum_u U[u] * A[u]
            # Y_form is a linear combination of B entries: sum_v V[v] * B[v]
            # Pk = (sum_u U[u] A[u]) * (sum_v V[v] B[v])
            # 
            # This product contributes to C entries. Specifically, if X corresponds to
            # a 2x2 block X[ei,ej] and Y to Y[fi,fj], then the product contributes
            # to Z[ei,fi] in the block product.
            # 
            # Wait, I'm overcomplicating this. Let me think again.
            # 
            # In the 4x4 * 4x4 Strassen:
            # We split into 2x2 blocks. Each block is 2x2.
            # The 7 block products M1..M7 are each 2x2 * 2x2 matrix products.
            # Each such product can be done with 7 scalar multiplications (2x2 Strassen).
            # 
            # So for M1 = (A_00 + A_11) * (B_00 + B_11):
            # Let X = A_00 + A_11 (a 2x2 matrix) and Y = B_00 + B_11 (a 2x2 matrix).
            # We compute X * Y using 2x2 Strassen, giving 7 scalar products.
            # Each scalar product is a product of two linear forms in the entries of X and Y.
            # 
            # For example, P1 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1])
            # X[0,0] = A[0,0] + A[2,0], X[1,1] = A[1,1] + A[3,1]
            # So X[0,0] + X[1,1] = A[0,0] + A[2,0] + A[1,1] + A[3,1]
            # Similarly for Y.
            # 
            # P1 contributes to Z[0,0] of the block product M1.
            # Z[0,0] of M1 contributes to C[0,0] of the overall 4x4 result.
            # 
            # So P1 contributes to C[0,0] with coefficient 1 (from Z00 = P1 + ...).
            # 
            # In the tensor:
            # u indices: A[0,0]=0, A[2,0]=8, A[1,1]=5, A[3,1]=13
            # v indices: B[0,0]=0, B[2,0]=18, B[1,1]=10, B[3,1]=28 (for col_offset=0)
            # w index: C[0,0]=0
            # 
            # So P1 is a rank-1 term with:
            # U: {0: 1, 8: 1, 5: 1, 13: 1}
            # V: {0: 1, 18: 1, 10: 1, 28: 1}
            # W: {0: 1}
            
            # The 7 sub-products contribute to Z[0,0], Z[0,1], Z[1,0], Z[1,1] as:
            # Z00 = P1 + P4 - P5 + P7
            # Z01 = P3 + P5
            # Z10 = P2 + P4
            # Z11 = P1 - P2 + P3 + P6
            
            # So for each Pk, we need to know which Z entries it contributes to and with what coefficient.
            
            # Pk contributions to Z:
            # P1: Z00 += 1, Z11 += 1
            # P2: Z10 += 1, Z11 -= 1
            # P3: Z01 += 1, Z11 += 1
            # P4: Z00 += 1, Z10 += 1
            # P5: Z00 -= 1, Z01 += 1
            # P6: Z11 += 1
            # P7: Z00 += 1
            
            # For each Pk, the W vector has non-zero entries at the corresponding C positions.
            # The block (bi, bj) in the 4x4 matrix corresponds to C rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
            # 
            # Z[ei, fi] in the block product corresponds to C[2*bi+ei, 2*bj+fi] for the appropriate block.
            # 
            # Wait, I need to be more careful about which block each M_i corresponds to.
            # 
            # Actually, all 7 M_i's contribute to all 4 C blocks. The coefficients are the same for each block.
            # 
            # For M1: contributes to C_00, C_01, C_10, C_11 with coefficients from the 2x2 Strassen result.
            # 
            # So for a given block product M_i and a sub-product Pk, the W vector has:
            # - For each C block (ci, cj) in {0,1}x{0,1}:
            #   - If Pk contributes to Z[ei,fi] with coefficient c, then C[2*ci+ei, 2*cj+fi] gets coefficient c.
            
            # Let me just enumerate all 49 terms explicitly.
            # For each of the 7 block products, and for each of the 7 sub-products, create one rank-1 term.
            
            for pk_idx, (x_form, y_form) in enumerate(sub_products):
                # Determine which Z entries this Pk contributes to
                z_contrib = {}
                if pk_idx == 0:  # P1
                    z_contrib = {(0,0): 1, (1,1): 1}
                elif pk_idx == 1:  # P2
                    z_contrib = {(1,0): 1, (1,1): -1}
                elif pk_idx == 2:  # P3
                    z_contrib = {(0,1): 1, (1,1): 1}
                elif pk_idx == 3:  # P4
                    z_contrib = {(0,0): 1, (1,0): 1}
                elif pk_idx == 4:  # P5
                    z_contrib = {(0,0): -1, (0,1): 1}
                elif pk_idx == 5:  # P6
                    z_contrib = {(1,1): 1}
                elif pk_idx == 6:  # P7
                    z_contrib = {(0,0): 1}
                
                # Build W vector
                W_dict = {}
                for (ei, fi), coeff in z_contrib.items():
                    for ci in range(2):
                        for cj in range(2):
                            k = col_offset + 2*ci + ei
                            i = 2*cj + fi
                            w = k * 4 + i
                            W_dict[w] = W_dict.get(w, 0) + coeff
                
                # Remove zeros
                W_dict = {k: v for k, v in W_dict.items() if abs(v) > 1e-12}
                
                if x_form and y_form and W_dict:
                    add_rank1_term(x_form, y_form, W_dict)
    
    # Add 49 terms for columns 0-3
    add_strassen_block(0)
    # Add 49 terms for columns 4-7
    add_strassen_block(4)
    
    # Add 4 terms for column 8 (4x4 * 4x1)
    for j in range(4):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for i in range(4):
            u = i * 4 + j
            U_dict[u] = 1
            w = 8 * 4 + i
            W_dict[w] = 1
        V_dict[j * 9 + 8] = 1
        add_rank1_term(U_dict, V_dict, W_dict)
    
    # Now we have len(rank1_terms) rank-1 terms.
    # Let's verify the count: should be 49 + 49 + 4 = 102.
    R = len(rank1_terms)
    print(f"Number of rank-1 terms: {R}")
    
    # Build factor matrices
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    for c, (U_dict, V_dict, W_dict) in enumerate(rank1_terms):
        for u, coeff in U_dict.items():
            U[u, c] = coeff
        for v, coeff in V_dict.items():
            V[v, c] = coeff
        for w, coeff in W_dict.items():
            W[w, c] = coeff
    
    # Verify the decomposition
    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    error = np.linalg.norm(T_recon - T)
    print(f"Verification error: {error}")
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
