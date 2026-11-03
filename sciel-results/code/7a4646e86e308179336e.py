import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use two 4x4x4 Strassen decompositions
    # Rank of 4x4x4 Strassen is 49
    # Total rank = 98
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen block
    # offset_u, offset_v, offset_w are index offsets
    # block_u, block_v, block_w are the 2x2 block indices
    # This is getting complex to map directly.
    # Let's generate the 49 terms for 4x4x4 explicitly.
    
    # Strassen 2x2x2 terms
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will build the 49 terms for 4x4x4
    # Each term corresponds to a product of two 2x2 matrices
    # The 4x4 matrices are split into four 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # The 7 block multiplications are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each of these is a 2x2x2 multiplication
    # We need to map the 2x2x2 decomposition to the 4x4x4 indices
    
    # Let's define the 2x2x2 Strassen decomposition
    # U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    # Rows of U_2x2: (i, j) for i,j in 0..1
    # Rows of V_2x2: (j, k) for j,k in 0..1
    # Rows of W_2x2: (k, i) for k,i in 0..1
    
    # Term 1: (A00+A11)(B00+B11)
    # U: A00 (1,1), A11 (1,1) -> U[0,0]=1, U[3,3]=1?
    # Wait, U indices are i*m+j. For 2x2, m=2.
    # A00 is i=0, j=0. A11 is i=1, j=1.
    # So U[0,0]=1, U[3,3]=1.
    # V: B00 (1,1), B11 (1,1).
    # V[0,0]=1, V[3,3]=1.
    # W: C00, C01, C10, C11.
    # C00 = M1 + ...
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's just hardcode the 49 terms for 4x4x4
    # This is tedious but safe.
    
    # Alternative: Use a library or known decomposition?
    # No, must be self-contained.
    
    # Let's write a function to generate the Strassen 4x4x4 decomposition
    def generate_strassen_4x4x4():
        terms = []
        # Each term is a tuple (u_dict, v_dict, w_dict)
        # u_dict maps (i, j) to coeff
        # v_dict maps (j, k) to coeff
        # w_dict maps (k, i) to coeff
        
        # 7 block multiplications
        # M1: (A00+A11)(B00+B11)
        # A00: i=0, j=0. A11: i=1, j=1.
        # B00: j=0, k=0. B11: j=1, k=1.
        # u: {(0,0):1, (1,1):1}
        # v: {(0,0):1, (1,1):1}
        # w: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to expand each block multiplication into 7 terms of 2x2x2
        # This is getting complicated.
        
        # Let's just use the 98 rank solution with simple block structure
        # If we can't easily generate Strassen, we can use a simpler lower rank?
        # No, 128 is the schoolbook.
        # Let's try to generate the Strassen terms programmatically.
        
        # Strassen 2x2x2 terms (7 terms)
        # Term 1: (A00+A11)(B00+B11)
        # Term 2: (A10+A11)B00
        # Term 3: A00(B10-B11)
        # Term 4: A11(B01-B00)
        # Term 5: (A00+A01)B11
        # Term 6: (A10-A00)(B00+B01)
        # Term 7: (A01-A11)(B10+B11)
        
        # For 4x4x4, we have 7 such block products.
        # Each block product is a 2x2x2 multiplication.
        # So we have 7 * 7 = 49 terms.
        
        # Let's define the 2x2x2 Strassen decomposition
        # U2 (4x7), V2 (4x7), W2 (4x7)
        # Rows of U2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Rows of V2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Rows of W2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Term 1: (A00+A11)(B00+B11)
        # U2: A00 (0,0)=1, A11 (1,1)=1 -> U2[0,0]=1, U2[3,0]=1
        # V2: B00 (0,0)=1, B11 (1,1)=1 -> V2[0,0]=1, V2[3,0]=1
        # W2: C00 (0,0)=1, C01 (0,1)=0, C10 (1,0)=0, C11 (1,1)=1
        # Wait, W2 maps to C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So for Term 1 (M1):
        # W2[0,0] = 1 (C00)
        # W2[1,0] = 0 (C01)
        # W2[2,0] = 0 (C10)
        # W2[3,0] = 1 (C11)
        
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # Term 2: (A10+A11)B00
        # U2: A10 (1,0)=1, A11 (1,1)=1 -> U2[2,1]=1, U2[3,1]=1
        # V2: B00 (0,0)=1 -> V2[0,1]=1
        # W2: C10 (1,0)=1, C11 (1,1)=-1
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        # Term 3: A00(B10-B11)
        # U2: A00 (0,0)=1 -> U2[0,2]=1
        # V2: B10 (1,0)=1, B11 (1,1)=-1 -> V2[2,2]=1, V2[3,2]=-1
        # W2: C01 (0,1)=1, C11 (1,1)=1
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        # Term 4: A11(B01-B00)
        # U2: A11 (1,1)=1 -> U2[3,3]=1
        # V2: B01 (0,1)=1, B00 (0,0)=-1 -> V2[1,3]=1, V2[0,3]=-1
        # W2: C00 (0,0)=1, C10 (1,0)=1
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        # Term 5: (A00+A01)B11
        # U2: A00 (0,0)=1, A01 (0,1)=1 -> U2[0,4]=1, U2[1,4]=1
        # V2: B11 (1,1)=1 -> V2[3,4]=1
        # W2: C00 (0,0)=-1, C01 (0,1)=1
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        # Term 6: (A10-A00)(B00+B01)
        # U2: A10 (1,0)=1, A00 (0,0)=-1 -> U2[2,5]=1, U2[0,5]=-1
        # V2: B00 (0,0)=1, B01 (0,1)=1 -> V2[0,5]=1, V2[1,5]=1
        # W2: C11 (1,1)=1
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # Term 7: (A01-A11)(B10+B11)
        # U2: A01 (0,1)=1, A11 (1,1)=-1 -> U2[1,6]=1, U2[3,6]=-1
        # V2: B10 (1,0)=1, B11 (1,1)=1 -> V2[2,6]=1, V2[3,6]=1
        # W2: C00 (0,0)=1
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        # Now we have U2, V2, W2 for 2x2x2
        # We need to map this to 4x4x4
        # The 4x4 matrices are split into 2x2 blocks
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # The 7 block multiplications are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each block multiplication, we use the 2x2x2 decomposition
        # But the indices are shifted.
        
        # Let's define the block offsets
        # A00: i=0, j=0. Offset 0.
        # A01: i=0, j=1. Offset 2 (j offset).
        # A10: i=1, j=0. Offset 2 (i offset).
        # A11: i=1, j=1. Offset 4 (i+j offset).
        
        # Actually, the row index in U is i*m+j.
        # For 4x4, m=4.
        # A00: i=0, j=0..1. Rows 0,1.
        # A01: i=0, j=2..3. Rows 2,3.
        # A10: i=1, j=0..1. Rows 4,5.
        # A11: i=1, j=2..3. Rows 6,7.
        
        # Similarly for V (j*p+k, p=4 for the block)
        # B00: j=0, k=0..1. Rows 0,1.
        # B01: j=0, k=2..3. Rows 2,3.
        # B10: j=1, k=0..1. Rows 4,5.
        # B11: j=1, k=2..3. Rows 6,7.
        
        # And W (k*n+i, n=4)
        # C00: k=0, i=0..1. Rows 0,1.
        # C01: k=0, i=2..3. Rows 2,3.
        # C10: k=1, i=0..1. Rows 4,5.
        # C11: k=1, i=2..3. Rows 6,7.
        
        # We will create the 49 terms
        # Term index t = block_idx * 7 + local_idx
        # block_idx in 0..6, local_idx in 0..6
        
        terms_4x4 = []
        for block_idx in range(7):
            # Determine the blocks involved in this multiplication
            # M1: A00+A11, B00+B11
            # M2: A10+A11, B00
            # M3: A00, B10-B11
            # M4: A11, B01-B00
            # M5: A00+A01, B11
            # M6: A10-A00, B00+B01
            # M7: A01-A11, B10+B11
            
            if block_idx == 0:
                a_blocks = [(0, 1), (1, 1)] # A00, A11
                b_blocks = [(0, 0), (1, 1)] # B00, B11
                a_coeffs = [1, 1]
                b_coeffs = [1, 1]
            elif block_idx == 1:
                a_blocks = [(1, 0), (1, 1)] # A10, A11
                b_blocks = [(0, 0)] # B00
                a_coeffs = [1, 1]
                b_coeffs = [1]
            elif block_idx == 2:
                a_blocks = [(0, 0)] # A00
                b_blocks = [(1, 0), (1, 1)] # B10, B11
                a_coeffs = [1]
                b_coeffs = [1, -1]
            elif block_idx == 3:
                a_blocks = [(1, 1)] # A11
                b_blocks = [(0, 1), (0, 0)] # B01, B00
                a_coeffs = [1]
                b_coeffs = [1, -1]
            elif block_idx == 4:
                a_blocks = [(0, 0), (0, 1)] # A00, A01
                b_blocks = [(1, 1)] # B11
                a_coeffs = [1, 1]
                b_coeffs = [1]
            elif block_idx == 5:
                a_blocks = [(1, 0), (0, 0)] # A10, A00
                b_blocks = [(0, 0), (0, 1)] # B00, B01
                a_coeffs = [1, -1]
                b_coeffs = [1, 1]
            elif block_idx == 6:
                a_blocks = [(0, 1), (1, 1)] # A01, A11
                b_blocks = [(1, 0), (1, 1)] # B10, B11
                a_coeffs = [1, -1]
                b_coeffs = [1, 1]
            
            # Now for each local term in 2x2x2
            for local_idx in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Get the 2x2x2 coefficients for this local term
                # U2[local_idx] is a vector of 4
                # V2[local_idx] is a vector of 4
                # W2[local_idx] is a vector of 4
                
                # We need to map the 2x2 indices to 4x4 indices
                # The 2x2 indices are (i2, j2), (j2, k2), (k2, i2)
                # i2, j2, k2 in 0..1
                
                # For A blocks:
                # The block is specified by (i_block, j_block)
                # The local indices (i2, j2) are added to the block indices
                # i = i_block + i2
                # j = j_block + j2
                
                # But we have multiple a_blocks?
                # No, the Strassen block multiplication combines the blocks.
                # e.g. M1 = (A00+A11)(B00+B11)
                # This is a single 2x2x2 multiplication of the combined matrices.
                # The combined matrix A' has A'_{i2, j2} = A00_{i2, j2} + A11_{i2, j2}
                # So U2[local_idx] gives the coefficients for A'_{i2, j2}
                # We need to distribute this to A00 and A11.
                
                # So for each (i2, j2) with coeff c:
                # For each a_block (i_b, j_b) with coeff a_c:
                # i = i_b + i2, j = j_b + j2
                # u_dict[(i, j)] += c * a_c
                
                # Similarly for B.
                
                for i2 in range(2):
                    for j2 in range(2):
                        c_u = U2[i2 * 2 + j2, local_idx]
                        if c_u != 0:
                            for idx, (i_b, j_b) in enumerate(a_blocks):
                                a_c = a_coeffs[idx]
                                i = i_b + i2
                                j = j_b + j2
                                u_dict[(i, j)] = u_dict.get((i, j), 0) + c_u * a_c
                
                for j2 in range(2):
                    for k2 in range(2):
                        c_v = V2[j2 * 2 + k2, local_idx]
                        if c_v != 0:
                            for idx, (j_b, k_b) in enumerate(b_blocks):
                                b_c = b_coeffs[idx]
                                j = j_b + j2
                                k = k_b + k2
                                v_dict[(j, k)] = v_dict.get((j, k), 0) + c_v * b_c
                
                for k2 in range(2):
                    for i2 in range(2):
                        c_w = W2[k2 * 2 + i2, local_idx]
                        if c_w != 0:
                            # W maps to C blocks
                            # The C blocks are determined by the multiplication
                            # M1 -> C00, C01, C10, C11
                            # The W2 indices (k2, i2) map to C_{k_b, i_b}
                            # But which C blocks?
                            # For M1, it contributes to all 4 C blocks.
                            # C00: k_b=0, i_b=0
                            # C01: k_b=0, i_b=1
                            # C10: k_b=1, i_b=0
                            # C11: k_b=1, i_b=1
                            # So k = k_b + k2, i = i_b + i2
                            # We need to know which C blocks are involved.
                            # For M1, all 4.
                            # For M2, C10, C11.
                            # For M3, C01, C11.
                            # For M4, C00, C10.
                            # For M5, C00, C01.
                            # For M6, C11.
                            # For M7, C00.
                            
                            # Let's define the C blocks for each M
                            if block_idx == 0:
                                c_blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]
                            elif block_idx == 1:
                                c_blocks = [(1, 0), (1, 1)]
                            elif block_idx == 2:
                                c_blocks = [(0, 1), (1, 1)]
                            elif block_idx == 3:
                                c_blocks = [(0, 0), (1, 0)]
                            elif block_idx == 4:
                                c_blocks = [(0, 0), (0, 1)]
                            elif block_idx == 5:
                                c_blocks = [(1, 1)]
                            elif block_idx == 6:
                                c_blocks = [(0, 0)]
                            
                            for k_b, i_b in c_blocks:
                                k = k_b + k2
                                i = i_b + i2
                                w_dict[(k, i)] = w_dict.get((k, i), 0) + c_w
                
                terms_4x4.append((u_dict, v_dict, w_dict))
        
        return terms_4x4
    
    terms_4x4 = generate_strassen_4x4x4()
    
    # Now map to 4x4x8
    # First 49 terms: k=0..3
    # Next 49 terms: k=4..7
    
    for t_idx in range(49):
        u_dict, v_dict, w_dict = terms_4x4[t_idx]
        
        # First block (k=0..3)
        for (i, j), val in u_dict.items():
            U[i * m + j][t_idx] = val
        
        for (j, k), val in v_dict.items():
            V[j * p + k][t_idx] = val
        
        for (k, i), val in w_dict.items():
            W[k * n + i][t_idx] = val
        
        # Second block (k=4..7)
        t_idx2 = t_idx + 49
        for (i, j), val in u_dict.items():
            U[i * m + j][t_idx2] = val
        
        for (j, k), val in v_dict.items():
            # k is 0..3, we need k+4
            V[j * p + (k + 4)][t_idx2] = val
        
        for (k, i), val in w_dict.items():
            # k is 0..3, we need k+4
            W[(k + 4) * n + i][t_idx2] = val
    
    return {"U": U, "V": V, "W": W}

