import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition for 4x4x4 using Strassen (rank 49)
    # and add schoolbook for the 5th column (rank 16).
    # Total rank 65.
    
    # 1. Define 2x2 Strassen coefficients
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # 
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # Coefficients for A in each M
    # M1: a00:1, a11:1
    # M2: a10:1, a11:1
    # M3: a00:1
    # M4: a11:1
    # M5: a00:1, a01:1
    # M6: a10:1, a00:-1
    # M7: a01:1, a11:-1
    
    A_coeffs_2x2 = [
        {0: 1, 3: 1},      # M1: a00, a11 (indices 0,1,2,3 for 00,01,10,11)
        {2: 1, 3: 1},      # M2: a10, a11
        {0: 1},            # M3: a00
        {3: 1},            # M4: a11
        {0: 1, 1: 1},      # M5: a00, a01
        {2: 1, 0: -1},     # M6: a10, a00
        {1: 1, 3: -1}      # M7: a01, a11
    ]
    
    B_coeffs_2x2 = [
        {0: 1, 3: 1},      # M1: b00, b11
        {0: 1},            # M2: b00
        {1: 1, 3: -1},     # M3: b01, b11
        {2: 1, 0: -1},     # M4: b10, b00
        {3: 1},            # M5: b11
        {0: 1, 2: 1},      # M6: b00, b10
        {1: 1, 3: 1}       # M7: b01, b11
    ]
    
    # Reconstruction coefficients for C in 2x2 Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_recon_2x2 = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00
        {2: 1, 4: 1},                # C01
        {1: 1, 3: 1},                # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]
    
    # 2. Apply to 4x4
    # 4x4 Strassen steps
    # M1_4 = (A00+A11)(B00+B11)
    # M2_4 = (A10+A11)B00
    # M3_4 = A00(B01-B11)
    # M4_4 = A11(B10-B00)
    # M5_4 = (A00+A01)B11
    # M6_4 = (A10-A00)(B00+B10)
    # M7_4 = (A01-A11)(B01+B11)
    
    # A_coeffs_4x4 for each M_k_4 (keys are block indices 0,1,2,3)
    A_coeffs_4x4 = [
        {0: 1, 3: 1},      # M1_4
        {2: 1, 3: 1},      # M2_4
        {0: 1},            # M3_4
        {3: 1},            # M4_4
        {0: 1, 1: 1},      # M5_4
        {2: 1, 0: -1},     # M6_4
        {1: 1, 3: -1}      # M7_4
    ]
    
    B_coeffs_4x4 = [
        {0: 1, 3: 1},      # M1_4
        {0: 1},            # M2_4
        {1: 1, 3: -1},     # M3_4
        {2: 1, 0: -1},     # M4_4
        {3: 1},            # M5_4
        {0: 1, 2: 1},      # M6_4
        {1: 1, 3: 1}       # M7_4
    ]
    
    C_recon_4x4 = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00
        {2: 1, 4: 1},                # C01
        {1: 1, 3: 1},                # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_coeff, v_coeff, w_coeff):
        # u_coeff: dict { (i,j): val } for A
        # v_coeff: dict { (j,k): val } for B
        # w_coeff: dict { (k,i): val } for C
        # We need to convert to flat indices
        u_flat = [0.0] * (n*m)
        v_flat = [0.0] * (m*p)
        w_flat = [0.0] * (p*n)
        
        for (i, j), val in u_coeff.items():
            u_flat[i*m + j] = val
        for (j, k), val in v_coeff.items():
            v_flat[j*p + k] = val
        for (k, i), val in w_coeff.items():
            w_flat[k*n + i] = val
            
        U.append(u_flat)
        V.append(v_flat)
        W.append(w_flat)
        
    # Generate 4x4 Strassen terms
    # For each of the 7 M_k_4:
    #   We have A_block_comb and B_block_comb.
    #   Each is a 2x2 matrix of linear combinations of A/B scalars.
    #   We decompose the product of these two 2x2 matrices using 2x2 Strassen.
    
    for k4 in range(7):
        # Get the 2x2 matrices X and Y (as linear combos of A/B scalars)
        # X is from A_coeffs_4x4[k4]
        # Y is from B_coeffs_4x4[k4]
        
        # We need to form the 7 sub-terms of 2x2 Strassen for X*Y
        for k2 in range(7):
            # Determine the A-side linear combination for this sub-term
            # The A-side of the 2x2 Strassen term k2 is a linear combo of the 4 entries of X.
            # X entries are linear combos of A scalars.
            
            # Let's build the u_coeff dict
            u_coeff = {}
            v_coeff = {}
            
            # A-side
            # A_coeffs_2x2[k2] gives weights for X entries (0,1,2,3)
            # X entries are determined by A_coeffs_4x4[k4]
            # X is a 2x2 matrix.
            # X[0,0] corresponds to A block 0? No.
            # A_coeffs_4x4[k4] maps block indices to weights.
            # The block index 0 is A00, 1 is A01, 2 is A10, 3 is A11.
            # So X is formed by taking the linear combination of the 4x4 blocks?
            # No, X is a 2x2 matrix where each entry is a linear combination of the *scalars* of A?
            # Yes.
            
            # Let's compute the scalar coefficients for A.
            # For each scalar A[i,j], what is its coefficient in the A-side of sub-term k2?
            
            # X is a 2x2 matrix.
            # X[r,c] = sum_{block} weight_{block} * A_block[r,c]
            # Wait, A_coeffs_4x4[k4] gives weights for the *blocks*.
            # So X[r,c] = sum_{b in A_coeffs_4x4[k4]} A_coeffs_4x4[k4][b] * A_b[r,c]
            
            # Now, the 2x2 Strassen term k2 uses a linear combination of X's entries.
            # A_coeffs_2x2[k2] gives weights for X[0,0], X[0,1], X[1,0], X[1,1].
            
            # So, coeff of A[i,j] in the A-side of sub-term k2 is:
            # sum_{x_row, x_col} A_coeffs_2x2[k2].get(x_row*2+x_col, 0) * 
            #   (sum_{b} A_coeffs_4x4[k4].get(b, 0) * (1 if A_b has i,j else 0))
            
            # A_b has i,j if:
            # b=0 (A00): i in 0..1, j in 0..1
            # b=1 (A01): i in 0..1, j in 2..3
            # b=2 (A10): i in 2..3, j in 0..1
            # b=3 (A11): i in 2..3, j in 2..3
            
            for i in range(n):
                for j in range(m):
                    total_coeff = 0.0
                    # Iterate over X entries (0..3)
                    for x_idx in range(4):
                        x_row = x_idx // 2
                        x_col = x_idx % 2
                        w_x = A_coeffs_2x2[k2].get(x_idx, 0)
                        if w_x == 0: continue
                        
                        # Sum over blocks
                        for b, w_b in A_coeffs_4x4[k4].items():
                            # Check if A_b contains A[i,j]
                            # b=0: i<2, j<2
                            # b=1: i<2, j>=2
                            # b=2: i>=2, j<2
                            # b=3: i>=2, j>=2
                            if b == 0 and i < 2 and j < 2:
                                # A[i,j] is A00[i,j]
                                # X[x_row, x_col] uses A00[x_row, x_col]
                                # So we need x_row == i%2 and x_col == j%2
                                if x_row == i % 2 and x_col == j % 2:
                                    total_coeff += w_x * w_b
                            elif b == 1 and i < 2 and j >= 2:
                                if x_row == i % 2 and x_col == (j-2) % 2:
                                    total_coeff += w_x * w_b
                            elif b == 2 and i >= 2 and j < 2:
                                if x_row == (i-2) % 2 and x_col == j % 2:
                                    total_coeff += w_x * w_b
                            elif b == 3 and i >= 2 and j >= 2:
                                if x_row == (i-2) % 2 and x_col == (j-2) % 2:
                                    total_coeff += w_x * w_b
                            
                    if total_coeff != 0:
                        u_coeff[(i, j)] = total_coeff
                        
            # Similarly for B-side
            for j in range(m):
                for k in range(4): # Only first 4 columns for now
                    total_coeff = 0.0
                    for y_idx in range(4):
                        y_row = y_idx // 2
                        y_col = y_idx % 2
                        w_y = B_coeffs_2x2[k2].get(y_idx, 0)
                        if w_y == 0: continue
                        
                        for b, w_b in B_coeffs_4x4[k4].items():
                            if b == 0 and j < 2 and k < 2:
                                if y_row == j % 2 and y_col == k % 2:
                                    total_coeff += w_y * w_b
                            elif b == 1 and j < 2 and k >= 2:
                                if y_row == j % 2 and y_col == (k-2) % 2:
                                    total_coeff += w_y * w_b
                            elif b == 2 and j >= 2 and k < 2:
                                if y_row == (j-2) % 2 and y_col == k % 2:
                                    total_coeff += w_y * w_b
                            elif b == 3 and j >= 2 and k >= 2:
                                if y_row == (j-2) % 2 and y_col == (k-2) % 2:
                                    total_coeff += w_y * w_b
                            
                    if total_coeff != 0:
                        v_coeff[(j, k)] = total_coeff
                        
            # Now for W-side
            # The result of the 2x2 Strassen term k2 contributes to C blocks.
            # C_recon_4x4[k4] gives weights for the 4 C blocks.
            # C_recon_2x2[k2] gives weights for the 4 entries of the 2x2 result of M_k4.
            
            # Wait, the 2x2 Strassen computes the product X*Y.
            # The result is a 2x2 matrix Z.
            # Z entries are linear combos of the 7 scalar products.
            # C_recon_2x2[k2] is NOT the reconstruction.
            # The 7 scalar products are the terms.
            # Each term contributes to the final C.
            
            # How does term (k4, k2) contribute to C?
            # It is the k2-th scalar product in the Strassen decomposition of X*Y.
            # The result Z is the 2x2 matrix product.
            # The entries of Z are linear combinations of the 7 scalar products.
            # Let R[c_idx] be the coefficient of the k2-th scalar product in Z[c_idx].
            
            # From 2x2 Strassen:
            # Z00 = M1 + M4 - M5 + M7
            # Z01 = M3 + M5
            # Z10 = M2 + M4
            # Z11 = M1 - M2 + M3 + M6
            
            # So R is determined by k2.
            R = [0.0, 0.0, 0.0, 0.0]
            # M1 is k2=0, M2 is k2=1, etc.
            if k2 == 0: R = [1, 0, 0, 1]
            elif k2 == 1: R = [0, 0, 1, -1]
            elif k2 == 2: R = [0, 1, 0, 1]
            elif k2 == 3: R = [1, 0, 1, 0]
            elif k2 == 4: R = [-1, 1, 0, 0]
            elif k2 == 5: R = [0, 0, 0, 1]
            elif k2 == 6: R = [1, 0, 0, 0]
            
            # Now, Z is the 2x2 matrix of C blocks.
            # C_block[r,c] gets contribution R[r*2+c] * (value of Z[r,c])?
            # No, Z[r,c] IS the value.
            # And C_block[r,c] is a linear combination of the 7 M_k4.
            # C_recon_4x4[k4] gives the weight of M_k4 in C_block.
            
            # So, for each C block (b_c in 0..3):
            # Weight from 4x4 level: w_c4 = C_recon_4x4[k4].get(b_c, 0)
            # Weight from 2x2 level: w_c2 = R[b_c] (since b_c corresponds to Z index)
            # Total weight: w_total = w_c4 * w_c2
            
            # This weight applies to the 2x2 block C_b_c.
            # C_b_c contains C[i,j] for specific i,j.
            
            for b_c in range(4):
                w_c4 = C_recon_4x4[k4].get(b_c, 0)
                w_c2 = R[b_c]
                w_total = w_c4 * w_c2
                if w_total == 0: continue
                
                # Map b_c to i,j ranges
                # b_c=0: i<2, j<2
                # b_c=1: i<2, j>=2
                # b_c=2: i>=2, j<2
                # b_c=3: i>=2, j>=2
                
                for i in range(n):
                    for j in range(4):
                        if b_c == 0 and i < 2 and j < 2:
                            # C[i,j] is in C00
                            # Z index for C00 is 0
                            # R[0] is the weight
                            # So w_total is the weight
                            w_coeff = {}
                            # We need to add w_total to W[(j, i)]?
                            # W index is k*n + i. Here k=j (0..3), i is row.
                            # So W[j*4 + i] += w_total
                            if (j, i) not in w_coeff:
                                w_coeff[(j, i)] = 0
                            w_coeff[(j, i)] += w_total
                        elif b_c == 1 and i < 2 and j >= 2:
                            # C[i,j] in C01. Z index 1.
                            # R[1]
                            if (j, i) not in w_coeff:
                                w_coeff[(j, i)] = 0
                            w_coeff[(j, i)] += w_total
                        elif b_c == 2 and i >= 2 and j < 2:
                            # C[i,j] in C10. Z index 2.
                            # R[2]
                            if (j, i) not in w_coeff:
                                w_coeff[(j, i)] = 0
                            w_coeff[(j, i)] += w_total
                        elif b_c == 3 and i >= 2 and j >= 2:
                            # C[i,j] in C11. Z index 3.
                            # R[3]
                            if (j, i) not in w_coeff:
                                w_coeff[(j, i)] = 0
                            w_coeff[(j, i)] += w_total

            if u_coeff and v_coeff and w_coeff:
                add_term(u_coeff, v_coeff, w_coeff)

    # Add schoolbook terms for 5th column (k=4)
    for i in range(n):
        for j in range(m):
            u_coeff = {(i, j): 1.0}
            v_coeff = {(j, 4): 1.0}
            w_coeff = {(4, i): 1.0}
            add_term(u_coeff, v_coeff, w_coeff)

    # Convert lists to numpy arrays or keep as lists?
    # Prompt asks for {'U', 'V', 'W'} with shapes.
    # Lists of lists are fine.
    
    return {"U": U, "V": V, "W": W}
