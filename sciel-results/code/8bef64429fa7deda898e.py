import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 4x4x9
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A B1 + A B2 + A B3
    
    # 1. Generate 2x2x2 Strassen decomposition
    # Returns U (4x7), V (4x7), W (4x7)
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))
    
    # Strassen formulas
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # Term 1: (a+d)(e+h)
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    # Contributes to C00 (+1), C11 (+1), C01 (-1), C10 (-1)
    W_22[0, 0] = 1; W_22[3, 0] = 1; W_22[1, 0] = -1; W_22[2, 0] = -1
    
    # Term 2: (c+d)e
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    # Contributes to C10 (+1), C11 (-1)
    W_22[2, 1] = 1; W_22[3, 1] = -1
    
    # Term 3: a(f-h)
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    # Contributes to C01 (+1), C11 (+1)
    W_22[1, 2] = 1; W_22[3, 2] = 1
    
    # Term 4: d(g-f)
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[1, 3] = -1
    # Contributes to C00 (+1), C10 (+1)
    W_22[0, 3] = 1; W_22[2, 3] = 1
    
    # Term 5: (a+b)h
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    # Contributes to C00 (-1), C01 (+1)
    W_22[0, 4] = -1; W_22[1, 4] = 1
    
    # Term 6: (c-a)(e+f)
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    # Contributes to C11 (+1)
    W_22[3, 5] = 1
    
    # Term 7: (b-d)(g+h)
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    # Contributes to C00 (+1), C11 (-1)
    W_22[0, 6] = 1; W_22[3, 6] = -1
    
    # 2. Generate 4x4x4 Strassen decomposition
    # We use 7 "block" multiplications.
    # Each block multiplication is 2x2x2.
    # Total rank 49.
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Define the linear combinations for the 7 block products
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # The 7 products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map the 2x2 Strassen U,V,W to the 4x4 context.
    # For each product Pk, we have a 2x2x2 Strassen decomposition.
    # The U,V vectors for the 2x2 case are linear forms in the 2x2 inputs.
    # Here, the 2x2 inputs are linear forms in the 4x4 inputs.
    
    # Let's define the linear forms for A and B for each Pk.
    # A indices in 4x4: 
    # 0:00, 1:01, 2:02, 3:03
    # 4:10, 5:11, 6:12, 7:13
    # 8:20, 9:21, 10:22, 11:23
    # 12:30, 13:31, 14:32, 15:33
    # Block A00: 0,1,4,5
    # Block A01: 2,3,6,7
    # Block A10: 8,9,12,13
    # Block A11: 10,11,14,15
    
    # We will iterate through the 7 products.
    for k in range(7):
        # Determine the linear forms for A and B for this product
        # u_A is a vector of length 16 (coeffs for A)
        # u_B is a vector of length 16 (coeffs for B)
        
        # Actually, we can just compose the 2x2 U,V with the block linear forms.
        # Let L_A be a 4x16 matrix where each row is the linear form for one entry of the 2x2 A-block.
        # Let L_B be a 4x16 matrix for B-block.
        
        # For P1: A_block = A00+A11, B_block = B00+B11
        # The 2x2 Strassen U vectors are applied to the 2x2 A_block.
        # So the 4x4 U vector is L_A @ U_22_col.
        
        # Let's define L_A and L_B for each k.
        
        if k == 0: # P1
            # A_block = A00 + A11
            # A00: 0,1,4,5. A11: 10,11,14,15
            L_A = np.zeros((4, 16))
            L_A[0, 0] = 1; L_A[0, 4] = 1 # A00[0,0] + A11[0,0]? No.
            # The 2x2 A_block is:
            # [ A00[0,0]+A11[0,0], A00[0,1]+A11[0,1] ]
            # [ A00[1,0]+A11[1,0], A00[1,1]+A11[1,1] ]
            # Indices in 4x4:
            # A00: 0(0,0), 1(0,1), 4(1,0), 5(1,1)
            # A11: 10(0,0), 11(0,1), 14(1,0), 15(1,1)
            L_A[0, 0] = 1; L_A[0, 10] = 1
            L_A[0, 1] = 1; L_A[0, 11] = 1
            L_A[1, 4] = 1; L_A[1, 14] = 1
            L_A[1, 5] = 1; L_A[1, 15] = 1
            
            L_B = np.zeros((4, 16))
            L_B[0, 0] = 1; L_B[0, 10] = 1
            L_B[0, 1] = 1; L_B[0, 11] = 1
            L_B[1, 4] = 1; L_B[1, 14] = 1
            L_B[1, 5] = 1; L_B[1, 15] = 1
            
            # Coefficients for C blocks
            # C00 += P1, C11 += P1, C01 -= P1, C10 -= P1
            # We will handle C mapping later.
            
        elif k == 1: # P2
            # A_block = A10 + A11
            # B_block = B00
            L_A = np.zeros((4, 16))
            L_A[0, 8] = 1; L_A[0, 10] = 1
            L_A[0, 9] = 1; L_A[0, 11] = 1
            L_A[1, 12] = 1; L_A[1, 14] = 1
            L_A[1, 13] = 1; L_A[1, 15] = 1
            
            L_B = np.zeros((4, 16))
            L_B[0, 0] = 1
            L_B[0, 1] = 1
            L_B[1, 4] = 1
            L_B[1, 5] = 1
            
        elif k == 2: # P3
            # A_block = A00
            # B_block = B01 - B11
            L_A = np.zeros((4, 16))
            L_A[0, 0] = 1
            L_A[0, 1] = 1
            L_A[1, 4] = 1
            L_A[1, 5] = 1
            
            L_B = np.zeros((4, 16))
            L_B[0, 2] = 1; L_B[0, 10] = -1
            L_B[0, 3] = 1; L_B[0, 11] = -1
            L_B[1, 6] = 1; L_B[1, 14] = -1
            L_B[1, 7] = 1; L_B[1, 15] = -1
            
        elif k == 3: # P4
            # A_block = A11
            # B_block = B10 - B01
            L_A = np.zeros((4, 16))
            L_A[0, 10] = 1
            L_A[0, 11] = 1
            L_A[1, 14] = 1
            L_A[1, 15] = 1
            
            L_B = np.zeros((4, 16))
            L_B[0, 8] = 1; L_B[0, 2] = -1
            L_B[0, 9] = 1; L_B[0, 3] = -1
            L_B[1, 12] = 1; L_B[1, 6] = -1
            L_B[1, 13] = 1; L_B[1, 7] = -1
            
        elif k == 4: # P5
            # A_block = A00 + A01
            # B_block = B11
            L_A = np.zeros((4, 16))
            L_A[0, 0] = 1; L_A[0, 2] = 1
            L_A[0, 1] = 1; L_A[0, 3] = 1
            L_A[1, 4] = 1; L_A[1, 6] = 1
            L_A[1, 5] = 1; L_A[1, 7] = 1
            
            L_B = np.zeros((4, 16))
            L_B[0, 10] = 1
            L_B[0, 11] = 1
            L_B[1, 14] = 1
            L_B[1, 15] = 1
            
        elif k == 5: # P6
            # A_block = A10 - A00
            # B_block = B00 + B01
            L_A = np.zeros((4, 16))
            L_A[0, 8] = 1; L_A[0, 0] = -1
            L_A[0, 9] = 1; L_A[0, 1] = -1
            L_A[1, 12] = 1; L_A[1, 4] = -1
            L_A[1, 13] = 1; L_A[1, 5] = -1
            
            L_B = np.zeros((4, 16))
            L_B[0, 0] = 1; L_B[0, 2] = 1
            L_B[0, 1] = 1; L_B[0, 3] = 1
            L_B[1, 4] = 1; L_B[1, 6] = 1
            L_B[1, 5] = 1; L_B[1, 7] = 1
            
        elif k == 6: # P7
            # A_block = A01 - A11
            # B_block = B10 + B11
            L_A = np.zeros((4, 16))
            L_A[0, 2] = 1; L_A[0, 10] = -1
            L_A[0, 3] = 1; L_A[0, 11] = -1
            L_A[1, 6] = 1; L_A[1, 14] = -1
            L_A[1, 7] = 1; L_A[1, 15] = -1
            
            L_B = np.zeros((4, 16))
            L_B[0, 8] = 1; L_B[0, 10] = 1
            L_B[0, 9] = 1; L_B[0, 11] = 1
            L_B[1, 12] = 1; L_B[1, 14] = 1
            L_B[1, 13] = 1; L_B[1, 15] = 1
            
        # Now, for each of the 7 sub-terms in the 2x2 Strassen:
        for s in range(7):
            idx = k * 7 + s
            
            # U_44 column idx = L_A @ U_22[:, s]
            U_44[:, idx] = L_A @ U_22[:, s]
            V_44[:, idx] = L_B @ V_22[:, s]
            
            # W_44:
            # The 2x2 Strassen W_22[:, s] gives the coefficients for the 2x2 output Z_k.
            # Z_k is a 2x2 matrix.
            # We need to map Z_k to the 4x4 output C.
            # The mapping depends on k.
            
            # Z_k[0,0] contributes to C_block[k][0,0]
            # Z_k[0,1] contributes to C_block[k][0,1]
            # Z_k[1,0] contributes to C_block[k][1,0]
            # Z_k[1,1] contributes to C_block[k][1,1]
            
            # The coefficients for C_block are given by the Strassen combination rules.
            # Let's define the contribution of each Z_k to the 4 C blocks.
            
            # C00: P1(+), P4(+), P5(-), P7(+)
            # C01: P3(+), P5(+)
            # C10: P2(+), P4(+)
            # C11: P1(+), P2(-), P3(+), P6(+)
            
            # So for a given k, which C blocks does it contribute to, and with what sign?
            contrib = np.zeros((4, 4)) # 4x4 matrix of coeffs for the 4 blocks
            
            if k == 0: # P1
                contrib[0, 0] = 1 # C00
                contrib[3, 3] = 1 # C11
                contrib[0, 1] = -1 # C01
                contrib[1, 0] = -1 # C10
            elif k == 1: # P2
                contrib[1, 0] = 1 # C10
                contrib[3, 3] = -1 # C11
            elif k == 2: # P3
                contrib[0, 1] = 1 # C01
                contrib[3, 3] = 1 # C11
            elif k == 3: # P4
                contrib[0, 0] = 1 # C00
                contrib[1, 0] = 1 # C10
            elif k == 4: # P5
                contrib[0, 0] = -1 # C00
                contrib[0, 1] = 1 # C01
            elif k == 5: # P6
                contrib[3, 3] = 1 # C11
            elif k == 6: # P7
                contrib[0, 0] = 1 # C00
                contrib[3, 3] = -1 # C11
            
            # W_22[:, s] is a vector of 4 coeffs for the 2x2 Z_k.
            # Z_k[0,0] * w0 + Z_k[0,1] * w1 + Z_k[1,0] * w2 + Z_k[1,1] * w3
            # We need to add these to the corresponding C blocks.
            
            # Z_k[0,0] is at block (0,0) of the 2x2 block structure?
            # No, Z_k is the entire 2x2 result.
            # The "block" in C is 2x2.
            # So Z_k[0,0] adds to C[0,0] (top-left of the block).
            # Z_k[0,1] adds to C[0,1] (top-right of the block).
            # Z_k[1,0] adds to C[1,0] (bottom-left of the block).
            # Z_k[1,1] adds to C[1,1] (bottom-right of the block).
            
            # We need to map the 4 entries of Z_k to the 16 entries of C.
            # The block for C is determined by k?
            # No, the Strassen algorithm sums the Z_k's to form C.
            # C is the sum of contributions from all P_k.
            # Each P_k is a 2x2 matrix.
            # P_k contributes to a specific 2x2 block of C?
            # No, P_k is one of the 7 intermediate matrices.
            # C is formed by linear combinations of P_k.
            # C = c00*P1 + c10*P2 ...
            # Wait, the standard Strassen for 2x2 blocks:
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            # Here C00, C01, C10, C11 are 2x2 blocks.
            # So P_k contributes to multiple blocks.
            
            # For a fixed k, P_k is a 2x2 matrix.
            # P_k[0,0] contributes to C_block[0,0][0,0], C_block[1,0][0,0], etc.
            # Specifically, if P_k contributes to C_block[i,j] with coeff c,
            # then P_k[0,0] contributes to C[2i, 2j].
            # P_k[0,1] contributes to C[2i, 2j+1].
            # P_k[1,0] contributes to C[2i+1, 2j].
            # P_k[1,1] contributes to C[2i+1, 2j+1].
            
            # So for each (i,j) in contrib, we add w * c to the corresponding 4 positions.
            
            w_vec = W_22[:, s]
            for i in range(2):
                for j in range(2):
                    c = contrib[i, j]
                    if c != 0:
                        # Map to 4x4 indices
                        # Block (i,j) starts at row 2i, col 2j
                        r0 = 2 * i
                        c0 = 2 * j
                        W_44[r0 * 4 + c0, idx] += c * w_vec[0]
                        W_44[r0 * 4 + c0 + 1, idx] += c * w_vec[1]
                        W_44[(r0 + 1) * 4 + c0, idx] += c * w_vec[2]
                        W_44[(r0 + 1) * 4 + c0 + 1, idx] += c * w_vec[3]
                        
    # 3. Generate 4x4x1 Schoolbook decomposition
    U_41 = np.zeros((16, 16))
    V_41 = np.zeros((16, 16))
    W_41 = np.zeros((4, 16)) # Output is 4x1
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            U_41[i * 4 + j, idx] = 1
            V_41[j, idx] = 1 # B is 4x1, index j
            W_41[i, idx] = 1
    
    # 4. Combine
    # B1 is cols 0-3 of B (4x9)
    # B2 is cols 4-7 of B
    # B3 is col 8 of B
    
    # U is 16 x (49+49+16)
    # V is 36 x (49+49+16)
    # W is 36 x (49+49+16)
    
    rank = 49 + 49 + 16
    U_total = np.zeros((16, rank))
    V_total = np.zeros((36, rank))
    W_total = np.zeros((36, rank))
    
    # Part 1: A B1
    U_total[:, 0:49] = U_44
    # V for B1: B1 is 4x4, part of 4x9.
    # B1 indices in 4x9: rows 0-3, cols 0-3.
    # V_44 is 16x49. We need to map to 36x49.
    V_part1 = np.zeros((36, 49))
    for r in range(4):
        for c in range(4):
            V_part1[r * 9 + c, :] = V_44[r * 4 + c, :]
    V_total[:, 0:49] = V_part1
    
    # W for C: C is 4x9.
    # C1 is 4x4, cols 0-3.
    W_part1 = np.zeros((36, 49))
    for r in range(4):
        for c in range(4):
            W_part1[r * 9 + c, :] = W_44[r * 4 + c, :]
    W_total[:, 0:49] = W_part1
    
    # Part 2: A B2
    U_total[:, 49:98] = U_44
    # V for B2: cols 4-7.
    V_part2 = np.zeros((36, 49))
    for r in range(4):
        for c in range(4):
            V_part2[r * 9 + (4 + c), :] = V_44[r * 4 + c, :]
    V_total[:, 49:98] = V_part2
    
    # W for C2: cols 4-7.
    W_part2 = np.zeros((36, 49))
    for r in range(4):
        for c in range(4):
            W_part2[r * 9 + (4 + c), :] = W_44[r * 4 + c, :]
    W_total[:, 49:98] = W_part2
    
    # Part 3: A B3
    U_total[:, 98:114] = U_41
    # V for B3: col 8.
    V_part3 = np.zeros((36, 16))
    for r in range(4):
        V_part3[r * 9 + 8, :] = V_41[r, :]
    V_total[:, 98:114] = V_part3
    
    # W for C3: col 8.
    W_part3 = np.zeros((36, 16))
    for r in range(4):
        W_part3[r * 9 + 8, :] = W_41[r, :]
    W_total[:, 98:114] = W_part3
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
