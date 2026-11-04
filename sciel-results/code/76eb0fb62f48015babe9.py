import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # --- Part 1: Strassen for 4x4x4 (Rank 49) ---
    
    # 2x2x2 Strassen Decomposition (Rank 7)
    # U2: (4, 7) indices (i,j)
    # V2: (4, 7) indices (j,k)
    # W2: (4, 7) indices (k,i)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00, A11
    U2[0, 0] = 1; U2[3, 0] = 1
    # B part: B00, B11
    V2[0, 0] = 1; V2[3, 0] = 1
    # C part: C00, C11 (M1 adds to C00 and C11)
    # Wait, W2 should be the "internal" Strassen W.
    # The internal W2 for M1 (which is a 2x2 product) distributes to C00, C01, C10, C11.
    # But M1 itself is a 2x2 matrix.
    # The decomposition of M1 = A_eff * B_eff uses the standard 2x2x2 tensor.
    # So W2 for column 0 is the standard 2x2x2 W for the product.
    # The "distribution" of M1 to the final 4x4 C is handled later when mapping W.
    
    # Let's define the 7 internal Strassen factors for a generic 2x2x2 product.
    # Col 0: M1 = (a+d)(e+h)
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, 0, 0, 1] # C00 gets M1? No.
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So the internal W2 for each column c' represents the contribution of that M_c' to the 2x2 result.
    # For c'=0 (M1): contributes +1 to C00, +1 to C11.
    W2[:, 0] = [1, 0, 0, 1] # Indices (k,i). C00 is k=0,i=0. C11 is k=1,i=1.
    
    # Col 1: M2 = (c+d)e
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 0, 1, 0] # C10 (k=1, i=0) gets M2. C11 (k=1, i=1) gets -M2.
    # Wait, W2 should hold the coefficients.
    # C10 = M2 + M4. So M2 contributes +1 to C10.
    # C11 = M1 - M2 + M3 + M6. So M2 contributes -1 to C11.
    W2[0, 1] = 0 # C00
    W2[1, 1] = 0 # C01
    W2[2, 1] = 1 # C10 (k=1, i=0)
    W2[3, 1] = -1 # C11 (k=1, i=1)
    
    # Col 2: M3 = a(f-h)
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 1, 0, -1]
    W2[0, 2] = 0
    W2[1, 2] = 1 # C01
    W2[2, 2] = 0
    W2[3, 2] = 1 # C11
    
    # Col 3: M4 = d(g-e)
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [0, 0, 1, -1]
    W2[0, 3] = 1 # C00
    W2[1, 3] = 0
    W2[2, 3] = 1 # C10
    W2[3, 3] = 0
    
    # Col 4: M5 = (a+b)h
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[0, 4] = -1 # C00
    W2[1, 4] = 1 # C01
    W2[2, 4] = 0
    W2[3, 4] = 0
    
    # Col 5: M6 = (c-a)(e+f)
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[0, 5] = 0
    W2[1, 5] = 0
    W2[2, 5] = 0
    W2[3, 5] = 1 # C11
    
    # Col 6: M7 = (b-d)(g+h)
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[0, 6] = 1 # C00
    W2[1, 6] = 0
    W2[2, 6] = 0
    W2[3, 6] = 0
    
    # Now build 4x4x4 Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block Mappings
    # A blocks: A11 (0-3), A12 (4-7), A21 (8-11), A22 (12-15)
    # B blocks: B11 (0-3), B12 (4-7), B21 (8-11), B22 (12-15)
    # C blocks: C11 (0-3), C12 (4-7), C21 (8-11), C22 (12-15)
    
    # Define the 7 block-level Strassen terms
    # Each term has:
    # A_map: list of (block_index, sign)
    # B_map: list of (block_index, sign)
    # C_map: list of (block_index, sign)
    
    terms = [
        # 1: M1 = (A11+A22)(B11+B22)
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)] },
        # 2: M2 = (A21+A22)B11
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)] },
        # 3: M3 = A11(B12-B22)
        { 'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)] },
        # 4: M4 = A22(B21-B11)
        { 'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)] },
        # 5: M5 = (A11+A12)B22
        { 'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)] },
        # 6: M6 = (A21-A11)(B11+B12)
        { 'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)] },
        # 7: M7 = (A12-A22)(B21+B22)
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)] },
    ]
    
    for l, term in enumerate(terms):
        base_col = l * 7
        for c_prime in range(7):
            col = base_col + c_prime
            
            # Fill U4
            for (b_idx, sign) in term['A']:
                # b_idx 0->A11 (0-3), 1->A12 (4-7), 2->A21 (8-11), 3->A22 (12-15)
                start = b_idx * 4
                U4[start:start+4, col] += sign * U2[:, c_prime]
                
            # Fill V4
            for (b_idx, sign) in term['B']:
                start = b_idx * 4
                V4[start:start+4, col] += sign * V2[:, c_prime]
                
            # Fill W4
            for (b_idx, sign) in term['C']:
                start = b_idx * 4
                W4[start:start+4, col] += sign * W2[:, c_prime]
                
    # --- Part 2: Schoolbook for 4x4x3 (Rank 48) ---
    
    U_sb = np.zeros((16, 48))
    V_sb = np.zeros((12, 48))
    W_sb = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                U_sb[i*4+j, col] = 1
                V_sb[j*3+k_prime, col] = 1
                # W index for C2: k_global = 4 + k_prime
                # W index = k_global * 4 + i
                W_sb[(4+k_prime)*4+i, col] = 1
                col += 1
                
    # --- Combine ---
    
    rank = 97
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    U[:, 0:49] = U4
    U[:, 49:97] = U_sb
    
    # V4 is 16x49 (for B1)
    # V_sb is 12x48 (for B2)
    V[0:16, 0:49] = V4
    V[16:28, 49:97] = V_sb
    
    # W4 is 16x49 (for C1)
    # W_sb is 12x48 (for C2)
    W[0:16, 0:49] = W4
    W[16:28, 49:97] = W_sb
    
    return {"U": U, "V": V, "W": W}

