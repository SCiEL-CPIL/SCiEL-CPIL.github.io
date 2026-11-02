import time
import random
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We will construct a decomposition of rank 81
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x2 using Schoolbook (rank 32)
    
    rank_total = 81
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]
    
    # Helper to add a rank-1 term to U, V, W
    def add_term(u_idx, v_idx, w_idx, c, u_val, v_val, w_val):
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val

    # Strassen decomposition for 2x2x2
    # Returns list of (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    # Local indices: u in {0,1,2,3}, v in {0,1,2,3}, w in {0,1,2,3}
    # u = i*2 + j, v = j*2 + k, w = k*2 + i
    def strassen_2x2x2():
        terms = []
        # P1 = (A11 + A22)(B11 + B22)
        # A11 is u=0, A22 is u=3
        # B11 is v=0, B22 is v=3
        # w contributions: C11 (w=0), C22 (w=3)
        
        # This is complex to map directly. 
        # Instead, let's just generate the 7 terms for a generic 2x2x2 mult
        # and handle the linear combinations of A and B in the caller?
        # No, the linear combinations are on the inputs A and B.
        # So for P1, the "effective" A is A11+A22 and "effective" B is B11+B22.
        # The tensor for P1 is T[i,j,k] where i,j,k are local 2x2 indices.
        # We decompose this T into 7 rank-1 terms.
        # Then we map the local indices to global indices.
        
        # Standard Strassen 2x2x2 decomposition (rank 7)
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = AB
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # Each mk is a product of a linear combo of A and a linear combo of B.
        # We need to expand these into rank-1 terms of the form U[i,j] V[j,k] W[k,i]
        
        # Let's define the 7 terms for 2x2x2
        # Term 1: m1 = (a+d)(e+h) = ae + ah + de + dh
        # This is a sum of 4 schoolbook terms.
        # So the rank of 2x2x2 is not 7 if we expand m1 into 4 terms?
        # Wait, Strassen's algorithm has rank 7 because the 7 products are computed.
        # But each product is a scalar multiplication of two 2x2 matrices? No.
        # In the tensor decomposition sense, the rank of the 2x2x2 tensor is 7.
        # This means we can write the tensor as a sum of 7 rank-1 tensors.
        # The "products" m1..m7 are the coefficients.
        
        # Let's find the explicit rank-1 terms for 2x2x2.
        # Reference: "The rank of the 2x2x2 tensor is 7".
        # One such decomposition is:
        # T = sum_{c=1}^7 U_c V_c W_c
        # where U_c, V_c, W_c are 4x1 vectors.
        
        # Let's use a known decomposition.
        # I will hardcode the 7 terms for 2x2x2.
        
        terms_2x2 = []
        # Indices: u: 00, 01, 10, 11 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
        # v: 00, 01, 10, 11 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
        # w: 00, 01, 10, 11 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1)
        
        # Term 1:
        u1 = [1, 0, 1, 0] # a, d
        v1 = [1, 1, 0, 0] # e, f
        w1 = [1, 0, 1, 0] # e, g? No.
        # This is getting risky to guess.
        
        # Alternative: Use schoolbook for 2x2x2 (rank 4) but we know rank is 7?
        # No, schoolbook is rank 4 for 2x2x2?
        # C11 = a e + b g
        # C12 = a f + b h
        # C21 = c e + d g
        # C22 = c f + d h
        # This is 4 products? No, 8 products.
        # Each product is a rank-1 tensor.
        # So schoolbook rank is 8.
        # Strassen rank is 7.
        
        # Let's just use schoolbook for the 2x2x2 blocks to be safe, and see if we can beat 96.
        # If we use schoolbook for 2x2x2, rank is 8.
        # Strassen blocks: 7 blocks.
        # Total rank = 7 * 8 = 56 for the 4x4x4 part.
        # Plus 32 for 4x4x2.
        # Total 88.
        # 88 < 96.
        
        # So even with schoolbook 2x2x2, we get 88.
        
        # Let's implement schoolbook 2x2x2 decomposition.
        # It is easier and guaranteed correct.
        
        return terms_2x2

    # Let's implement the 88 rank solution.
    # Part 1: 4x4x4 via 7 schoolbook 2x2x2 decompositions.
    # Part 2: 4x4x2 via schoolbook.
    
    # Reset U, V, W for rank 88
    rank_total = 88
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]
    
    col = 0
    
    # Define Strassen linear combinations
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in 2x2 block grid
    # B blocks: B11, B12, B21, B22
    # C blocks: C11, C12, C21, C22
    
    # Strassen products:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # We will iterate through these 7 products.
    # For each product, we determine the "effective" A and B blocks.
    # Then we perform schoolbook multiplication for those 2x2 blocks.
    # The result is added to the appropriate C blocks with signs.
    
    def add_strassen_product(a_coeffs, b_coeffs, c_coeffs, sign, col_offset):
        # a_coeffs: dict mapping block index (0..3) to coefficient (1, -1, 0)
        # b_coeffs: dict mapping block index (0..3) to coefficient (1, -1, 0)
        # c_coeffs: dict mapping block index (0..3) to coefficient (1, -1, 0)
        # sign: global sign for this product's contribution
        # col_offset: starting column index
        
        # For each non-zero A block and B block
        for ai, a_val in a_coeffs.items():
            if a_val == 0: continue
            for bi, b_val in b_coeffs.items():
                if b_val == 0: continue
                
                # Determine global indices for this 2x2x2 multiplication
                # A block ai: row block ai//2, col block ai%2
                # B block bi: row block bi//2, col block bi%2
                # Check if they are compatible: A col block == B row block
                if ai % 2 != bi // 2:
                    continue # Should not happen for valid Strassen terms
                
                a_row_block = ai // 2
                a_col_block = ai % 2
                b_row_block = bi // 2
                b_col_block = bi % 2
                
                # The product contributes to C block (a_row_block, b_col_block)
                c_block = a_row_block * 2 + b_col_block
                
                c_val = c_coeffs.get(c_block, 0)
                if c_val == 0:
                    continue
                
                # Now we have a valid 2x2x2 multiplication
                # A sub-block at (a_row_block, a_col_block)
                # B sub-block at (b_row_block, b_col_block)
                # C sub-block at (a_row_block, b_col_block)
                
                # Schoolbook for this 2x2x2
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            # Global indices
                            gi = a_row_block * 2 + i
                            gj = a_col_block * 2 + j
                            gk = b_col_block * 2 + k
                            
                            u_idx = gi * m + gj
                            v_idx = gj * p + gk
                            w_idx = gk * n + gi
                            
                            u_val = a_val
                            v_val = b_val
                            w_val = c_val * sign
                            
                            U[u_idx][col_offset] += u_val
                            V[v_idx][col_offset] += v_val
                            W[w_idx][col_offset] += w_val
                            
                            col_offset += 1
                            
        return col_offset

    # Define the 7 Strassen terms
    # Block indices: 00=0, 01=1, 10=2, 11=3
    # A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    
    # P1 = (A11 + A22)(B11 + B22)
    a1 = {0: 1, 3: 1}
    b1 = {0: 1, 3: 1}
    c1 = {0: 1, 3: 1} # C11 and C22?
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So P1 contributes to C11 (idx 0) and C22 (idx 3) with +1
    c1 = {0: 1, 3: 1}
    
    col = add_strassen_product(a1, b1, c1, 1, col)
    
    # P2 = (A21 + A22)B11
    a2 = {2: 1, 3: 1}
    b2 = {0: 1}
    # P2 contributes to C21 (idx 2) with +1
    c2 = {2: 1}
    
    col = add_strassen_product(a2, b2, c2, 1, col)
    
    # P3 = A11(B12 - B22)
    a3 = {0: 1}
    b3 = {1: 1, 3: -1}
    # P3 contributes to C12 (idx 1) with +1
    c3 = {1: 1}
    
    col = add_strassen_product(a3, b3, c3, 1, col)
    
    # P4 = A22(B21 - B11)
    a4 = {3: 1}
    b4 = {2: 1, 0: -1}
    # P4 contributes to C11 (idx 0) and C21 (idx 2) with +1
    c4 = {0: 1, 2: 1}
    
    col = add_strassen_product(a4, b4, c4, 1, col)
    
    # P5 = (A11 + A12)B22
    a5 = {0: 1, 1: 1}
    b5 = {3: 1}
    # P5 contributes to C11 (idx 0) with -1 and C12 (idx 1) with +1
    c5 = {0: -1, 1: 1}
    
    col = add_strassen_product(a5, b5, c5, 1, col)
    
    # P6 = (A21 - A11)(B11 + B12)
    a6 = {2: 1, 0: -1}
    b6 = {0: 1, 1: 1}
    # P6 contributes to C22 (idx 3) with +1
    c6 = {3: 1}
    
    col = add_strassen_product(a6, b6, c6, 1, col)
    
    # P7 = (A12 - A22)(B21 + B22)
    a7 = {1: 1, 3: -1}
    b7 = {2: 1, 3: 1}
    # P7 contributes to C11 (idx 0) with +1
    c7 = {0: 1}
    
    col = add_strassen_product(a7, b7, c7, 1, col)
    
    # Now handle the 4x4x2 part (columns 4..5 of B)
    # B is 4x6. Columns 0-3 handled above.
    # Columns 4-5 are the "extra" part.
    # We can treat this as a 4x4x2 multiplication.
    # A is 4x4.
    # B_extra is 4x2.
    # C_extra is 4x2.
    
    # Schoolbook for 4x4x2
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # k is local index for the 2 columns
                # Global k is 4 + k
                gk = 4 + k
                
                u_idx = i * m + j
                v_idx = j * p + gk
                w_idx = gk * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Check if col == rank_total
    # It should be 88
    
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
