def generate_strassen_4x4_terms():
    terms = []
    
    def add_scalar_term(i, j, k, coeff):
        # i, j, k are indices in 0..3
        # Term is coeff * A[i, j] * B[j, k]
        terms.append((i, j, k, coeff))
        
    # We need to express the Strassen intermediate matrices as sums of A[i,j]
    # and B[j,k].
    
    # M1 = (A00 + A11) * (B00 + B11)
    # A00 is A[0:2, 0:2], A11 is A[2:4, 2:4]
    # B00 is B[0:2, 0:2], B11 is B[2:4, 2:4]
    
    # M1 terms:
    # (A[i,j] + A[i+2, j+2]) * (B[j,k] + B[j+2, k+2]) for i,j in 0,1
    # This expands to:
    # A[i,j]B[j,k] + A[i,j]B[j+2, k+2] + A[i+2, j+2]B[j,k] + A[i+2, j+2]B[j+2, k+2]
    
    # Let's list the 7 Strassen products and their contributions to C blocks.
    
    # Product 1: P1 = (A00 + A11)(B00 + B11)
    # Contributes to: C00 (+), C11 (-)
    
    # Product 2: P2 = (A10 + A11)B00
    # Contributes to: C10 (+), C11 (-)
    
    # Product 3: P3 = A00(B10 - B11)
    # Contributes to: C01 (+), C11 (+)
    
    # Product 4: P4 = A11(B01 - B00)
    # Contributes to: C00 (+), C10 (+)
    
    # Product 5: P5 = (A00 + A01)B11
    # Contributes to: C00 (-), C01 (+)
    
    # Product 6: P6 = (A10 - A00)(B00 + B01)
    # Contributes to: C11 (+)
    
    # Product 7: P7 = (A01 - A11)(B10 + B11)
    # Contributes to: C00 (-), C11 (-)
    
    # We need to expand each Pi into scalar terms A[i,j]B[j,k].
    
    # Helper to add terms for a product of two 2x2 blocks
    # block_a: (r, c) top-left of A block
    # block_b: (r, c) top-left of B block
    # coeff: multiplier for the whole product
    
    def add_block_product(a_r, a_c, b_r, b_c, coeff):
        for i in range(2):
            for j in range(2):
                # A index: a_r+i, a_c+j
                # B index: b_r+j, b_c+i  (Wait, B is 2x2, so row offset b_r, col offset b_c)
                # The product A_block * B_block results in a 2x2 matrix.
                # Element (u, v) of the product is sum_j A[u,j] B[j,v].
                # Here u,v in 0..1.
                # A global index: (a_r+u, a_c+j)
                # B global index: (b_r+j, b_c+v)
                # Wait, the term is A[i,j] * B[j,k].
                # So for each u,v in 0..1:
                #   sum_{j=0}^1 A[a_r+u, a_c+j] * B[b_r+j, b_c+v]
                
                for v in range(2):
                    for j in range(2):
                        add_scalar_term(a_r+u, a_c+j, b_r+j, b_c+v, coeff)
                        
    # This helper is wrong because Strassen combines A's and B's first.
    # P1 = (A00 + A11) * (B00 + B11)
    # This is (A00*B00 + A00*B11 + A11*B00 + A11*B11).
    # So we can just call add_block_product for each combination.
    
    # P1
    add_block_product(0, 0, 0, 0, 1) # A00*B00
    add_block_product(0, 0, 2, 2, 1) # A00*B11
    add_block_product(2, 2, 0, 0, 1) # A11*B00
    add_block_product(2, 2, 2, 2, 1) # A11*B11
    
    # P2 = (A10 + A11) * B00
    add_block_product(2, 0, 0, 0, 1) # A10*B00
    add_block_product(2, 2, 0, 0, 1) # A11*B00
    
    # P3 = A00 * (B10 - B11)
    add_block_product(0, 0, 2, 0, 1) # A00*B10
    add_block_product(0, 0, 2, 2, -1) # A00*B11
    
    # P4 = A11 * (B01 - B00)
    add_block_product(2, 2, 0, 2, 1) # A11*B01
    add_block_product(2, 2, 0, 0, -1) # A11*B00
    
    # P5 = (A00 + A01) * B11
    add_block_product(0, 0, 2, 2, 1) # A00*B11
    add_block_product(0, 2, 2, 2, 1) # A01*B11
    
    # P6 = (A10 - A00) * (B00 + B01)
    add_block_product(2, 0, 0, 0, 1) # A10*B00
    add_block_product(2, 0, 0, 2, 1) # A10*B01
    add_block_product(0, 0, 0, 0, -1) # A00*B00
    add_block_product(0, 0, 0, 2, -1) # A00*B01
    
    # P7 = (A01 - A11) * (B10 + B11)
    add_block_product(0, 2, 2, 0, 1) # A01*B10
    add_block_product(0, 2, 2, 2, 1) # A01*B11
    add_block_product(2, 2, 2, 0, -1) # A11*B10
    add_block_product(2, 2, 2, 2, -1) # A11*B11
    
    # Now we have a list of scalar terms for the 4x4 Strassen.
    # But we need to combine them according to the C block formulas.
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # The `terms` list currently contains terms for each Pi separately?
    # No, my `add_block_product` adds to a global `terms` list.
    # I need to track which Pi each term belongs to.
    
    # Let's restart the generation with proper tracking.
    
    return terms
