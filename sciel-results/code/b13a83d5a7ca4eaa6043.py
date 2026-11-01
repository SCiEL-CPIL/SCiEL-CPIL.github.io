"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    # Rank 49 via recursive Strassen: 7 block-level products, each with 7 scalar products
    
    # Strassen's 7 formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai* is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Strassen at block level:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 product, decomposed via Strassen into 7 scalar products.
    # Total rank = 7 * 7 = 49
    
    # Let's build the U, V, W matrices directly.
    # Indexing: A has indices (i, j) for i,j in 0..3
    # B has indices (j, k) for j,k in 0..3
    # C has indices (i, k) for i,k in 0..3
    # W[k*n+i, r] = W[k*4+i, r]
    
    # U[i*m+j, r] = U[i*4+j, r]
    # V[j*p+k, r] = V[j*4+k, r]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # For each block-level Strassen product Pi (i=0..6),
    # and for each scalar-level Strassen product (j=0..6),
    # we have one rank term.
    
    # Block-level Strassen: 7 products
    # For product Pi, we need to express:
    #   Pi = (sum of A blocks with coeffs) * (sum of B blocks with coeffs)
    # Then each 2x2 block product is decomposed via scalar Strassen.
    
    # Let me define the 7 block-level products:
    # Each product Pi = Xi * Yi where Xi is a linear combo of A-blocks, Yi is a linear combo of B-blocks
    
    # A-blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B-blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # Block-level Strassen products:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B10-B11)
    # P3 = A11(B01-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # For each Pi = Xi * Yi, where Xi and Yi are 2x2 matrices (linear combos of A/B blocks),
    # we decompose Xi * Yi using scalar Strassen.
    
    # Scalar Strassen for 2x2 matrices X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    # M0 = (x00+x11)(y00+y11)
    # M1 = (x10+x11)y00
    # M2 = x00(y10-y11)
    # M3 = x11(y01-y00)
    # M4 = (x00+x01)y11
    # M5 = (x10-x00)(y00+y01)
    # M6 = (x01-x11)(y10+y11)
    # Result: [[M0+M3-M4+M6, M2+M4], [M1+M3, M0-M1+M2+M5]]
    
    # For each block product Pi, Xi is a 2x2 matrix whose entries are linear combinations of A elements.
    # Similarly for Yi with B elements.
    
    # Let me define the block-level products more carefully.
    # For P0 = (A00+A11)(B00+B11):
    #   X = A00 + A11, Y = B00 + B11
    #   X is 2x2: X[0][0]=A[0][0]+A[2][2], X[0][1]=A[0][1]+A[2][3], X[1][0]=A[1][0]+A[3][2], X[1][1]=A[1][1]+A[3][3]
    #   Y is 2x2: Y[0][0]=B[0][0]+B[2][2], Y[0][1]=B[0][1]+B[2][3], Y[1][0]=B[1][0]+B[3][2], Y[1][1]=B[1][1]+B[3][3]
    
    # I'll define each block product by specifying:
    # - For each of the 4 entries of X (2x2), which A elements contribute and with what coefficient
    # - For each of the 4 entries of Y (2x2), which B elements contribute and with what coefficient
    
    # A element (i,j) corresponds to U index i*4+j
    # B element (j,k) corresponds to V index j*4+k
    
    # Block product definitions: (X_coeffs, Y_coeffs)
    # X_coeffs: list of 4 entries, each is list of (a_row, a_col, coeff)
    # Y_coeffs: list of 4 entries, each is list of (b_row, b_col, coeff)
    
    block_products = [
        # P0 = (A00+A11)(B00+B11)
        {
            'X': [
                [(0,0,1), (2,2,1)],  # X[0][0]
                [(0,1,1), (2,3,1)],  # X[0][1]
                [(1,0,1), (3,2,1)],  # X[1][0]
                [(1,1,1), (3,3,1)],  # X[1][1]
            ],
            'Y': [
                [(0,0,1), (2,2,1)],  # Y[0][0]
                [(0,1,1), (2,3,1)],  # Y[0][1]
                [(1,0,1), (3,2,1)],  # Y[1][0]
                [(1,1,1), (3,3,1)],  # Y[1][1]
            ]
        },
        # P1 = (A10+A11)B00
        {
            'X': [
                [(2,0,1), (2,2,1)],  # X[0][0] = A[2][0]+A[2][2]
                [(2,1,1), (2,3,1)],  # X[0][1] = A[2][1]+A[2][3]
                [(3,0,1), (3,2,1)],  # X[1][0] = A[3][0]+A[3][2]
                [(3,1,1), (3,3,1)],  # X[1][1] = A[3][1]+A[3][3]
            ],
            'Y': [
                [(0,0,1)],  # Y[0][0]
                [(0,1,1)],  # Y[0][1]
                [(1,0,1)],  # Y[1][0]
                [(1,1,1)],  # Y[1][1]
            ]
        },
        # P2 = A00(B10-B11)
        {
            'X': [
                [(0,0,1)],  # X[0][0]
                [(0,1,1)],  # X[0][1]
                [(1,0,1)],  # X[1][0]
                [(1,1,1)],  # X[1][1]
            ],
            'Y': [
                [(2,0,1), (2,2,-1)],  # Y[0][0]
                [(2,1,1), (2,3,-1)],  # Y[0][1]
                [(3,0,1), (3,2,-1)],  # Y[1][0]
                [(3,1,1), (3,3,-1)],  # Y[1][1]
            ]
        },
        # P3 = A11(B01-B00)
        {
            'X': [
                [(2,2,1)],  # X[0][0]
                [(2,3,1)],  # X[0][1]
                [(3,2,1)],  # X[1][0]
                [(3,3,1)],  # X[1][1]
            ],
            'Y': [
                [(0,0,-1), (0,1,1)],  # Y[0][0]
                [(0,1,1), (0,1,1)],   # Wait, let me redo this
                [(1,0,-1), (1,1,1)],  # Y[1][0]
                [(1,1,1), (1,1,1)],   # Hmm, need to be careful
            ]
        },
    ]
    
    # Let me restart with a cleaner approach. I'll directly compute the contributions.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 block-level Strassen products.
    # For each product, specify:
    #   X: 2x2 matrix of linear combos of A elements: X[r][c] = list of (i, j, coeff) meaning coeff * A[i][j]
    #   Y: 2x2 matrix of linear combos of B elements: Y[r][c] = list of (j, k, coeff) meaning coeff * B[j][k]
    
    # A indices: A[i][j], i=0..3, j=0..3. U index = i*4+j
    # B indices: B[j][k], j=0..3, k=0..3. V index = j*4+k
    # C indices: C[i][k], i=0..3, k=0..3. W index = k*4+i
    
    # Block products (Strassen at 4x4 level):
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B10-B11)
    # P3 = A11(B01-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # Where A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # Similarly for B.
    
    # Let me define each product's X and Y matrices explicitly.
    
    def make_X_from_blocks(blocks):
        """blocks: list of (block_id, coeff) where block_id is 'A00','A01','A10','A11'"""
        X = [[[] for _ in range(2)] for _ in range(2)]
        for block_id, coeff in blocks:
            if block_id == 'A00':
                X[0][0].append((0, 0, coeff))
                X[0][1].append((0, 1, coeff))
                X[1][0].append((1, 0, coeff))
                X[1][1].append((1, 1, coeff))
            elif block_id == 'A01':
                X[0][0].append((0, 2, coeff))
                X[0][1].append((0, 3, coeff))
                X[1][0].append((1, 2, coeff))
                X[1][1].append((1, 3, coeff))
            elif block_id == 'A10':
                X[0][0].append((2, 0, coeff))
                X[0][1].append((2, 1, coeff))
                X[1][0].append((3, 0, coeff))
                X[1][1].append((3, 1, coeff))
            elif block_id == 'A11':
                X[0][0].append((2, 2, coeff))
                X[0][1].append((2, 3, coeff))
                X[1][0].append((3, 2, coeff))
                X[1][1].append((3, 3, coeff))
        return X
    
    def make_Y_from_blocks(blocks):
        """blocks: list of (block_id, coeff) where block_id is 'B00','B01','B10','B11'"""
        Y = [[[] for _ in range(2)] for _ in range(2)]
        for block_id, coeff in blocks:
            if block_id == 'B00':
                Y[0][0].append((0, 0, coeff))
                Y[0][1].append((0, 1, coeff))
                Y[1][0].append((1, 0, coeff))
                Y[1][1].append((1, 1, coeff))
            elif block_id == 'B01':
                Y[0][0].append((0, 2, coeff))
                Y[0][1].append((0, 3, coeff))
                Y[1][0].append((1, 2, coeff))
                Y[1][1].append((1, 3, coeff))
            elif block_id == 'B10':
                Y[0][0].append((2, 0, coeff))
                Y[0][1].append((2, 1, coeff))
                Y[1][0].append((3, 0, coeff))
                Y[1][1].append((3, 1, coeff))
            elif block_id == 'B11':
                Y[0][0].append((2, 2, coeff))
                Y[0][1].append((2, 3, coeff))
                Y[1][0].append((3, 2, coeff))
                Y[1][1].append((3, 3, coeff))
        return Y
    
    # Define the 7 block products
    block_prods = [
        # P0 = (A00+A11)(B00+B11)
        (make_X_from_blocks([('A00',1),('A11',1)]), make_Y_from_blocks([('B00',1),('B11',1)])),
        # P1 = (A10+A11)B00
        (make_X_from_blocks([('A10',1),('A11',1)]), make_Y_from_blocks([('B00',1)])),
        # P2 = A00(B10-B11)
        (make_X_from_blocks([('A00',1)]), make_Y_from_blocks([('B10',1),('B11',-1)])),
        # P3 = A11(B01-B00)
        (make_X_from_blocks([('A11',1)]), make_Y_from_blocks([('B01',1),('B00',-1)])),
        # P4 = (A00+A01)B11
        (make_X_from_blocks([('A00',1),('A01',1)]), make_Y_from_blocks([('B11',1)])),
        # P5 = (A10-A00)(B00+B01)
        (make_X_from_blocks([('A10',1),('A00',-1)]), make_Y_from_blocks([('B00',1),('B01',1)])),
        # P6 = (A01-A11)(B10+B11)
        (make_X_from_blocks([('A01',1),('A11',-1)]), make_Y_from_blocks([('B10',1),('B11',1)])),
    ]
    
    # For each block product Pi with X and Y (each 2x2), decompose X*Y via scalar Strassen.
    # Scalar Strassen for X=[[x00,x01],[x10,x11]], Y=[[y00,y01],[y10,y11]]:
    # M0 = (x00+x11)(y00+y11)
    # M1 = (x10+x11)y00
    # M2 = x00(y10-y11)
    # M3 = x11(y01-y00)
    # M4 = (x00+x01)y11
    # M5 = (x10-x00)(y00+y01)
    # M6 = (x01-x11)(y10+y11)
    # C = [[M0+M3-M4+M6, M2+M4], [M1+M3, M0-M1+M2+M5]]
    
    # For each of the 7 scalar Strassen products, the "left" and "right" are linear combos of X/Y entries,
    # which are themselves linear combos of A/B elements.
    
    # I need to express each scalar Strassen product as:
    #   (sum of A elements with coeffs) * (sum of B elements with coeffs)
    # and add to U, V, W.
    
    # Let me define a helper to add a product of two linear combos.
    # left: list of (a_idx, coeff) where a_idx = i*4+j
    # right: list of (b_idx, coeff) where b_idx = j*4+k
    # For the product, each term contributes to C[i][k] with coeff_a * coeff_b
    
    col = 0
    for X, Y in block_prods:
        # X is 2x2, each entry is list of (i, j, coeff)
        # Y is 2x2, each entry is list of (j, k, coeff)
        
        # Define the 7 scalar Strassen products:
        # Each is (left_coeffs, right_coeffs) where left is combo of X entries, right is combo of Y entries
        # X entries: X[0][0], X[0][1], X[1][0], X[1][1]
        # Y entries: Y[0][0], Y[0][1], Y[1][0], Y[1][1]
        
        scalar_strassen = [
            # M0 = (x00+x11)(y00+y11)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M1 = (x10+x11)y00
            ([(1,0,1),(1,1,1)], [(0,0,1)]),
            # M2 = x00(y10-y11)
            ([(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3 = x11(y01-y00)
            ([(1,1,1)], [(0,1,1),(0,0,-1)]),
            # M4 = (x00+x01)y11
            ([(0,0,1),(0,1,1)], [(1,1,1)]),
            # M5 = (x10-x00)(y00+y01)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            # M6 = (x01-x11)(y10+y11)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ]
        
        # Result C entries:
        # C[0][0] = M0 + M3 - M4 + M6
        # C[0][1] = M2 + M4
        # C[1][0] = M1 + M3
        # C[1][1] = M0 - M1 + M2 + M5
        
        # For each scalar Strassen product, compute the left and right as linear combos of A/B elements.
        for left_coeffs, right_coeffs in scalar_strassen:
            # left_coeffs: list of (x_row, x_col, coeff) -> combine X entries
            # Build left as list of (a_i, a_j, coeff)
            left_elements = {}
            for xr, xc, coeff in left_coeffs:
                for ai, aj, c in X[xr][xc]:
                    key = (ai, aj)
                    left_elements[key] = left_elements.get(key, 0) + coeff * c
            
            # right_coeffs: list of (y_row, y_col, coeff) -> combine Y entries
            right_elements = {}
            for yr, yc, coeff in right_coeffs:
                for bj, bk, c in Y[yr][yc]:
                    key = (bj, bk)
                    right_elements[key] = right_elements.get(key, 0) + coeff * c
            
            # Now for each (a_i, a_j) in left and (b_j2, b_k) in right,
            # the product contributes to C[a_i][b_k] with coeff_left * coeff_right
            # But we need to add this as a rank term to U, V, W.
            
            # U[a_i*4+a_j, col] += coeff_left
            # V[b_j2*4+b_k, col] += coeff_right
            # W[b_k*4+a_i, col] += 1 (since the product is left*right, and we sum over j)
            
            # Wait, the tensor decomposition requires:
            # C[i][k] = sum_r U[i*4+j_r... ] 
            # Actually: C[i][k] = sum_r U[i*m + j, r] * V[j*p + k, r] ... no.
            
            # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # where T[i,j,k] = 1 if this is a valid matrix mult term (always 1 for our case, 
            # since C[i][k] = sum_j A[i][j]*B[j][k])
            
            # So for each rank term r, we have U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
            # and the product U*V*W should equal 1 for the (i,j,k) that this term corresponds to.
            
            # In our Strassen decomposition, each rank term corresponds to a specific product
            # of a linear combo of A elements and a linear combo of B elements, contributing to C.
            
            # Let me think about this differently. The standard approach:
            # For a rank-1 term with U vector u, V vector v, W vector w:
            #   u has nonzeros at positions (i, j) for various (i,j)
            #   v has nonzeros at positions (j, k) for various (j,k)
            #   w has nonzeros at positions (k, i) for various (k,i)
            # The contribution to C[i][k] is sum_j u[i,j] * v[j,k] * w[k,i]
            
            # For Strassen, each rank term is a product of two linear forms.
            # Let left = sum_{(i,j)} a_{ij} * A[i][j]
            # Let right = sum_{(j,k)} b_{jk} * B[j][k]
            # Then left * right = sum_{i,j,k} a_{ij} * b_{jk} * A[i][j] * B[j][k]
            # This contributes to C[i][k] the term sum_j a_{ij} * b_{jk}
            
            # So for this rank term:
            # U[i*4+j, col] = a_{ij} for all (i,j) in left
            # V[j*4+k, col] = b_{jk} for all (j,k) in right
            # W[k*4+i, col] = 1 for all (i,k) that appear in the result
            
            # But W should be such that the product U*V*W gives the right C contribution.
            # C[i][k] += sum_j U[i*4+j, col] * V[j*4+k, col] * W[k*4+i, col]
            # We want this to equal sum_j a_{ij} * b_{jk}
            
            # If we set W[k*4+i, col] = 1, then we get sum_j a_{ij} * b_{jk} * 1 = sum_j a_{ij} * b_{jk}. 
            # That's correct!
            
            # So for each rank term:
            # U[i*4+j, col] = a_{ij}
            # V[j*4+k, col] = b_{jk}
            # W[k*4+i, col] = 1 for all (i,k) in the support
            
            # But wait, W[k*4+i, col] = 1 for ALL (i,k) in the support of the result.
            # The result C[i][k] gets contribution from all (i,j,k) where a_{ij} and b_{jk} are nonzero.
            
            # Let me collect all (i,k) pairs that appear.
            
            ik_pairs = set()
            for (ai, aj), ca in left_elements.items():
                if ca == 0:
                    continue
                for (bj, bk), cb in right_elements.items():
                    if cb == 0:
                        continue
                    if aj == bj:  # This is the valid matrix mult index
                        ik_pairs.add((ai, bk))
            
            # Now set U, V, W
            for (ai, aj), ca in left_elements.items():
                if ca != 0:
                    U[ai * 4 + aj][col] += ca
            
            for (bj, bk), cb in right_elements.items():
                if cb != 0:
                    V[bj * 4 + bk][col] += cb
            
            for (ai, bk) in ik_pairs:
                W[bk * 4 + ai][col] += 1
            
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
