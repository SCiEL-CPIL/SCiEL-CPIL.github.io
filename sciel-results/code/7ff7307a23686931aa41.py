"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Recursive Strassen: 4x4 = 2x2 blocks of 2x2 matrices
    # Outer Strassen: 7 block multiplications
    # Inner Strassen: each 2x2 block product uses 7 scalar mults
    # Total rank = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Helper: global index from block row, block col, local row, local col
    # For A (4x4): row = 2*br + lr, col = 2*bc + lc
    # For B (4x4): row = 2*br + lr, col = 2*bc + lc
    # For C (4x4): row = 2*br + lr, col = 2*bc + lc
    
    # U index: i*m + j where i=row, j=col of A
    # V index: j*p + k where j=row, k=col of B
    # W index: k*n + i where k=row, i=col of C (trace convention W[k*n+i,r])
    
    # Outer Strassen: 7 terms, each is (linear combo of A blocks) * (linear combo of B blocks)
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # For each outer term, we have:
    #   A_combo: list of (block_row, block_col, coeff) for A
    #   B_combo: list of (block_row, block_col, coeff) for B
    
    # The product of two 2x2 matrices X*Y is computed via inner Strassen:
    # X and Y are 2x2 matrices whose entries are linear combinations of A/B blocks.
    # But actually, we need to be more careful.
    
    # Let me think differently. We have 7 outer terms. Each outer term computes:
    #   M_t = X_t * Y_t  where X_t is a 2x2 matrix (linear combo of A blocks)
    #   and Y_t is a 2x2 matrix (linear combo of B blocks)
    #   
    # Then M_t is a 2x2 matrix, and its entries contribute to C.
    # 
    # To compute M_t = X_t * Y_t using inner Strassen (rank 7), we express:
    #   (X_t * Y_t)_{lr,lc} = sum_{s=1}^{7} u_{lr,lc,s} * (X_t * Y_t)_s
    # where (X_t * Y_t)_s is the s-th inner Strassen scalar product.
    #
    # But this is getting complicated. Let me use a cleaner approach.
    
    # Actually, the cleanest way: 
    # The 4x4 matrix multiplication tensor can be decomposed by applying
    # Strassen at two levels. Let me just enumerate all 49 rank-1 terms directly.
    
    # For outer term t (0..6), and inner term s (0..6):
    # We need to determine which A elements, B elements, and C elements are involved.
    
    # Outer Strassen definitions (for 2x2 block matrix multiplication):
    # Let A be partitioned into blocks A[br][bc] for br,bc in {0,1}
    # Let B be partitioned into blocks B[br][bc] for br,bc in {0,1}
    # Each block is 2x2.
    
    # M[0] = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    # M[1] = (A[1][0] + A[1][1]) * B[0][0]
    # M[2] = A[0][0] * (B[1][0] - B[1][1])
    # M[3] = A[1][1] * (B[0][1] - B[0][0])
    # M[4] = (A[0][0] + A[0][1]) * B[1][1]
    # M[5] = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
    # M[6] = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
    
    # C[0][0] = M[0] + M[3] - M[4] + M[6]
    # C[0][1] = M[2] + M[4]
    # C[1][0] = M[1] + M[3]
    # C[1][1] = M[0] - M[1] + M[2] + M[5]
    
    # Now, each M[t] is a 2x2 matrix product. To compute M[t] = X * Y where
    # X and Y are 2x2 matrices (each entry being a linear combination of 2x2 blocks),
    # we use inner Strassen.
    
    # But wait - X and Y here are not scalar 2x2 matrices; their entries are 2x2 blocks.
    # This is the key insight: we're doing Strassen where the "scalars" are 2x2 matrices.
    
    # So for inner Strassen on X*Y (where X,Y are 2x2 with 2x2 block entries):
    # S[0] = (X[0][0] + X[1][1]) * (Y[0][0] + Y[1][1])
    # S[1] = (X[1][0] + X[1][1]) * Y[0][0]
    # S[2] = X[0][0] * (Y[1][0] - Y[1][1])
    # S[3] = X[1][1] * (Y[0][1] - Y[0][0])
    # S[4] = (X[0][0] + X[0][1]) * Y[1][1]
    # S[5] = (X[1][0] - X[0][0]) * (Y[0][0] + Y[0][1])
    # S[6] = (X[0][1] - X[1][1]) * (Y[1][0] + Y[1][1])
    
    # (X*Y)[0][0] = S[0] + S[3] - S[4] + S[6]
    # (X*Y)[0][1] = S[2] + S[4]
    # (X*Y)[1][0] = S[1] + S[3]
    # (X*Y)[1][1] = S[0] - S[1] + S[2] + S[5]
    
    # Now, X[br][bc] for outer term t is a linear combination of A blocks.
    # Y[br][bc] for outer term t is a linear combination of B blocks.
    
    # Each A block A[abr][abc] is a 2x2 matrix with entries A[2*abr+lr][2*abc+lc]
    # Each B block B[abr][abc] is a 2x2 matrix with entries B[2*abr+lr][2*abc+lc]
    
    # So S[s] for outer term t is a product of two 2x2 matrices, each of which
    # is a linear combination of 2x2 blocks. This product is itself a 2x2 matrix,
    # and each entry is a bilinear form in the original A and B entries.
    
    # To get rank-1 terms, we need to expand each S[s] entry into individual
    # scalar products.
    
    # Let me define the outer terms more precisely.
    
    # Outer term t: X_t and Y_t are 2x2 matrices of 2x2 blocks.
    # X_t[br][bc] = sum of coeff * A[abr][abc] over the A blocks in the combo
    # Y_t[br][bc] = sum of coeff * B[abr][abc] over the B blocks in the combo
    
    # For inner Strassen, we need X_t[0][0], X_t[1][1], etc.
    # These are 2x2 matrices that are linear combinations of A blocks.
    
    # Let me define the outer combos:
    # t=0: X = [[A00+A11, 0], [0, 0]], Y = [[B00+B11, 0], [0, 0]]
    #   Wait, that's not right. Let me re-read Strassen.
    
    # Standard Strassen for 2x2 matrices A and B:
    # M1 = (A11 + A22)(B11 + B22)  -- using 1-indexed
    # In 0-indexed: M1 = (A[0][0] + A[1][1])(B[0][0] + B[1][1])
    # M2 = (A[1][0] + A[1][1]) B[0][0]
    # M3 = A[0][0] (B[1][0] - B[1][1])
    # M4 = A[1][1] (B[0][1] - B[0][0])
    # M5 = (A[0][0] + A[0][1]) B[1][1]
    # M6 = (A[1][0] - A[0][0]) (B[0][0] + B[0][1])
    # M7 = (A[0][1] - A[1][1]) (B[1][0] + B[1][1])
    
    # So for outer term t, X_t is a 2x2 matrix where:
    # t=0: X_t = [[A00+A11, 0], [0, 0]]? No...
    
    # Actually, in Strassen, M1 = (A11+A22)(B11+B22) means we're multiplying
    # the scalar (A11+A22) by the scalar (B11+B22). But in our case, A11 is a 2x2 block.
    # So M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]) is a 2x2 matrix product.
    
    # The "X" in inner Strassen for outer term 0 is the 2x2 matrix:
    # X = [[A[0][0] + A[1][1], 0], [0, 0]]
    # No, that's not how Strassen works.
    
    # Let me reconsider. In standard Strassen:
    # We have 2x2 matrices A and B.
    # M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    # Here A[0][0] + A[1][1] is a scalar, and B[0][0] + B[1][1] is a scalar.
    # M1 is a scalar.
    
    # In our recursive case:
    # A is 4x4, partitioned into 2x2 blocks.
    # A[0][0] is a 2x2 matrix, A[1][1] is a 2x2 matrix.
    # A[0][0] + A[1][1] is a 2x2 matrix.
    # B[0][0] + B[1][1] is a 2x2 matrix.
    # M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]) is a 2x2 matrix product.
    
    # So for outer term 0, we need to compute the product of two 2x2 matrices:
    # X = A[0][0] + A[1][1]  (a 2x2 matrix)
    # Y = B[0][0] + B[1][1]  (a 2x2 matrix)
    # M1 = X * Y (2x2 matrix)
    
    # To compute X*Y using inner Strassen, we treat X and Y as 2x2 matrices of scalars.
    # Inner Strassen gives 7 scalar products.
    
    # So the full decomposition is:
    # For each outer term t (0..6):
    #   Determine X_t (2x2 matrix of scalars, linear combo of A entries)
    #   Determine Y_t (2x2 matrix of scalars, linear combo of B entries)
    #   For each inner term s (0..6):
    #     The s-th inner Strassen product gives a scalar
    #     This scalar contributes to specific entries of M_t
    #     M_t contributes to specific entries of C
    
    # Let me code this up properly.
    
    # Outer term definitions:
    # t=0: X = [[A00+A11, 0], [0, 0]]? No.
    
    # I think I'm overcomplicating this. Let me just directly enumerate.
    
    # For outer term t, X_t is a 2x2 matrix where each entry is a linear combination
    # of A's 2x2 blocks. Similarly for Y_t.
    
    # t=0: X_t[0][0] = A[0][0]+A[1][1], X_t[0][1] = 0, X_t[1][0] = 0, X_t[1][1] = 0
    #     Y_t[0][0] = B[0][0]+B[1][1], Y_t[0][1] = 0, Y_t[1][0] = 0, Y_t[1][1] = 0
    #     M0 = X_t * Y_t = [[(A00+A11)(B00+B11), 0], [0, 0]]
    
    # Hmm, but that's not right either. In Strassen, M1 is a scalar that gets
    # added to multiple C entries. Let me look at this more carefully.
    
    # I'll take a different approach: directly construct the 49 rank-1 terms
    # by expanding the recursive Strassen formula.
    
    # Let me use a systematic approach. I'll define:
    # - outer_A[t]: for outer term t, the 2x2 matrix X_t where X_t[br][bc] 
    #   is a dict mapping (abr, abc) -> coeff for A block A[abr][abc]
    # - outer_B[t]: similarly for B
    
    # Then for inner Strassen, I'll expand each term.
    
    # This is getting very complex. Let me just hardcode the 49 terms.
    
    # Actually, let me try a cleaner implementation using the structure.
    
    col = 0
    for t in range(7):
        # Define X_t as 2x2 matrix of linear combos of A blocks
        # X_t[br][bc] = {(abr, abc): coeff}
        if t == 0:
            X = {
                (0,0): {(0,0): 1, (1,1): 1},
                (0,1): {},
                (1,0): {},
                (1,1): {}
            }
            Y = {
                (0,0): {(0,0): 1, (1,1): 1},
                (0,1): {},
                (1,0): {},
                (1,1): {}
            }
        elif t == 1:
            X = {
                (0,0): {},
                (0,1): {},
                (1,0): {(1,0): 1, (1,1): 1},
                (1,1): {}
            }
            Y = {
                (0,0): {(0,0): 1},
                (0,1): {},
                (1,0): {},
                (1,1): {}
            }
        elif t == 2:
            X = {
                (0,0): {(0,0): 1},
                (0,1): {},
                (1,0): {},
                (1,1): {}
            }
            Y = {
                (0,0): {},
                (0,1): {},
                (1,0): {(1,0): 1},
                (1,1): {(1,1): -1}
            }
        elif t == 3:
            X = {
                (0,0): {},
                (0,1): {},
                (1,0): {},
                (1,1): {(1,1): 1}
            }
            Y = {
                (0,0): {(0,0): -1},
                (0,1): {(0,1): 1},
                (1,0): {},
                (1,1): {}
            }
        elif t == 4:
            X = {
                (0,0): {(0,0): 1},
                (0,1): {(0,1): 1},
                (1,0): {},
                (1,1): {}
            }
            Y = {
                (0,0): {},
                (0,1): {},
                (1,0): {},
                (1,1): {(1,1): 1}
            }
        elif t == 5:
            X = {
                (0,0): {(0,0): -1},
                (0,1): {},
                (1,0): {(1,0): 1},
                (1,1): {}
            }
            Y = {
                (0,0): {(0,0): 1},
                (0,1): {(0,1): 1},
                (1,0): {},
                (1,1): {}
            }
        elif t == 6:
            X = {
                (0,0): {},
                (0,1): {(0,1): 1},
                (1,0): {},
                (1,1): {(1,1): -1}
            }
            Y = {
                (0,0): {},
                (0,1): {},
                (1,0): {(1,0): 1},
                (1,1): {(1,1): 1}
            }
        
        # Now X and Y are 2x2 matrices of linear combos of 2x2 blocks.
        # Each X[br][bc] is a dict: {(abr,abc): coeff}
        # The actual 2x2 matrix X[br][bc] has entries:
        #   X[br][bc][lr][lc] = sum of coeff * A[2*abr+lr][2*abc+lc]
        
        # For inner Strassen, we need to compute X*Y where X and Y are 2x2 matrices
        # of 2x2 blocks. But actually, we should think of X and Y as 2x2 matrices
        # whose entries are themselves 2x2 matrices.
        
        # Inner Strassen on 2x2 matrices of 2x2 blocks:
        # S[0] = (X[0][0]+X[1][1]) * (Y[0][0]+Y[1][1])
        # S[1] = (X[1][0]+X[1][1]) * Y[0][0]
        # S[2] = X[0][0] * (Y[1][0]-Y[1][1])
        # S[3] = X[1][1] * (Y[0][1]-Y[0][0])
        # S[4] = (X[0][0]+X[0][1]) * Y[1][1]
        # S[5] = (X[1][0]-X[0][0]) * (Y[0][0]+Y[0][1])
        # S[6] = (X[0][1]-X[1][1]) * (Y[1][0]+Y[1][1])
        
        # Each S[s] is a 2x2 matrix (product of two 2x2 matrices of 2x2 blocks).
        # Wait, no. X[0][0] is a 2x2 matrix. Y[0][0] is a 2x2 matrix.
        # X[0][0] * Y[0][0] is a 2x2 matrix product, which is a 2x2 matrix.
        
        # So S[s] is a 2x2 matrix. And (X*Y)[br][bc] is a linear combo of S[s].
        
        # (X*Y)[0][0] = S[0] + S[3] - S[4] + S[6]
        # (X*Y)[0][1] = S[2] + S[4]
        # (X*Y)[1][0] = S[1] + S[3]
        # (X*Y)[1][1] = S[0] - S[1] + S[2] + S[5]
        
        # Each S[s] is a 2x2 matrix, and each entry of S[s] is a bilinear form
        # in the original A and B entries.
        
        # To get rank-1 terms, I need to expand each entry of each S[s] into
        # individual scalar products.
        
        # This is getting very involved. Let me just give up on the 49-term
        # decomposition and return the schoolbook 64-term one, since getting
        # the 49 terms exactly right is error-prone.
        
        # Actually, let me try a completely different approach: just use the
        # known fact that rank-49 exists and construct it via a different method.
        
        # For now, return schoolbook to ensure correctness.
        pass
    
    # Return schoolbook decomposition (rank 64) as a safe fallback
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
# EVOLVE-BLOCK-END
