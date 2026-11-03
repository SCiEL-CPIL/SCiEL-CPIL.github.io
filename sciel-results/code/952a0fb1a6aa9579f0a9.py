import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4 x 4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4 x 4x1 multiplication using Schoolbook (Rank 16)
    
    total_rank = 49 + 16
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # Low-level Strassen vectors for 2x2 multiplication
    # Indices 0..3 correspond to [00, 01, 10, 11]
    u_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    v_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_strassen = [
        [1, 0, 0, 1], # C00, C01, C10, C11
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ]
    
    # High-level Strassen block definitions
    # Each entry is a 2x2 matrix of coefficients for the 4 blocks A00, A01, A10, A11
    # Block indices: 00, 01, 10, 11
    # A_S is a 2x2 matrix of these block-vectors?
    # Actually, A_S is a 2x2 matrix where each element is a sum of blocks.
    # We can represent the contribution of each global A entry to the local A_S entries.
    
    # Let's define the linear map from Global A (16 entries) to Local A_S (4 entries)
    # and Global B (16 entries) to Local B_S (4 entries)
    # for each S in 1..7.
    
    # Global A indices: i (0..3), j (0..3). Index = i*4 + j.
    # Block A_IJ corresponds to i in {2I, 2I+1}, j in {2J, 2J+1}.
    
    # For a given S, A_S is a 2x2 matrix.
    # A_S[0,0] = sum of coefficients * A_blocks
    # We need to map the local indices (0..3) of A_S to global indices.
    
    # High-level Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Define the block coefficients for A and B for each S.
    # coeff_A[S][local_idx] is a list of (block_idx, coeff)
    # local_idx 0..3 corresponds to A_S[0,0], A_S[0,1], A_S[1,0], A_S[1,1]
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    
    a_coeffs = [
        # S=1: A00 + A11 for both rows/cols?
        # M1 = (A00+A11)(B00+B11).
        # A_S is 2x2.
        # A_S[0,0] = A00+A11, A_S[0,1] = A00+A11, A_S[1,0] = A00+A11, A_S[1,1] = A00+A11.
        [[(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]],
        # S=2: (A10+A11)B00.
        # A_S[0,0] = A10+A11, A_S[0,1] = A10+A11, A_S[1,0] = A10+A11, A_S[1,1] = A10+A11.
        [[(2,1),(3,1)], [(2,1),(3,1)], [(2,1),(3,1)], [(2,1),(3,1)]],
        # S=3: A00(B01-B11).
        # A_S = A00 (all entries).
        [[(0,1)], [(0,1)], [(0,1)], [(0,1)]],
        # S=4: A11(B10-B00).
        # A_S = A11.
        [[(3,1)], [(3,1)], [(3,1)], [(3,1)]],
        # S=5: (A00+A01)B11.
        # A_S[0,0]=A00+A01, A_S[0,1]=A00+A01, A_S[1,0]=A00+A01, A_S[1,1]=A00+A01.
        [[(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)]],
        # S=6: (A10-A00)(B00+B01).
        # A_S = A10-A00.
        [[(2,1),(0,-1)], [(2,1),(0,-1)], [(2,1),(0,-1)], [(2,1),(0,-1)]],
        # S=7: (A01-A11)(B10+B11).
        # A_S = A01-A11.
        [[(1,1),(3,-1)], [(1,1),(3,-1)], [(1,1),(3,-1)], [(1,1),(3,-1)]]
    ]
    
    b_coeffs = [
        # S=1: B00+B11
        [[(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]],
        # S=2: B00
        [[(0,1)], [(0,1)], [(0,1)], [(0,1)]],
        # S=3: B01-B11
        [[(1,1),(3,-1)], [(1,1),(3,-1)], [(1,1),(3,-1)], [(1,1),(3,-1)]],
        # S=4: B10-B00
        [[(2,1),(0,-1)], [(2,1),(0,-1)], [(2,1),(0,-1)], [(2,1),(0,-1)]],
        # S=5: B11
        [[(3,1)], [(3,1)], [(3,1)], [(3,1)]],
        # S=6: B00+B01
        [[(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)]],
        # S=7: B10+B11
        [[(2,1),(3,1)], [(2,1),(3,1)], [(2,1),(3,1)], [(2,1),(3,1)]]
    ]
    
    # We need to map block coefficients to global U/V vectors.
    # For a given S and t:
    # We form the global U vector by summing contributions from A blocks.
    
    for S in range(7):
        for t in range(7):
            # Get local u_t, v_t, w_t
            u_t = u_strassen[t]
            v_t = v_strassen[t]
            w_t = w_strassen[t]
            
            # Construct global U vector (size 16)
            U_vec = np.zeros(16)
            # For each local index li in 0..3:
            for li in range(4):
                coeff = u_t[li]
                if coeff == 0: continue
                # Get block coeffs for A_S[li]
                for block_idx, b_coeff in a_coeffs[S][li]:
                    # Add b_coeff * coeff to the corresponding global indices
                    # Block block_idx (0..3) corresponds to:
                    # 0: A00 (i 0,1; j 0,1)
                    # 1: A01 (i 0,1; j 2,3)
                    # 2: A10 (i 2,3; j 0,1)
                    # 3: A11 (i 2,3; j 2,3)
                    I, J = divmod(block_idx, 2)
                    for i_l in range(2):
                        for j_l in range(2):
                            i = 2*I + i_l
                            j = 2*J + j_l
                            idx = i*4 + j
                            U_vec[idx] += coeff * b_coeff
            
            # Construct global V vector (size 20, but only first 16 used for 4x4 part)
            V_vec = np.zeros(20)
            for li in range(4):
                coeff = v_t[li]
                if coeff == 0: continue
                for block_idx, b_coeff in b_coeffs[S][li]:
                    I, J = divmod(block_idx, 2)
                    for i_l in range(2):
                        for j_l in range(2):
                            j = 2*I + i_l # Row index in B (m dimension)
                            k = 2*J + j_l # Col index in B (p dimension, 0..3)
                            idx = j*5 + k
                            V_vec[idx] += coeff * b_coeff
            
            # Construct global W vector (size 20)
            # The result of M_S is a 2x2 block matrix.
            # The local w_t gives coefficients for C_S[0,0], C_S[0,1], C_S[1,0], C_S[1,1].
            # C_S is a 2x2 block of the final C.
            # The block position of M_S in C?
            # M1 contributes to C00, C01, C10, C11?
            # Recall:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5 + M7
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # So M_S contributes to specific blocks of C.
            
            # We need to know which C blocks M_S contributes to.
            # This is encoded in the high-level Strassen formulas.
            # Let's define c_block_coeffs[S] which is a 2x2 matrix of coefficients for C blocks.
            
    # Define c_block_coeffs
    c_block_coeffs = [
        # S=1: M1
        [[1, 0], [0, 1]], # C00, C01, C10, C11?
        # Wait, C is 2x2 blocks.
        # C00 gets +M1. C11 gets +M1.
        # So C_S[0,0] coeff 1, C_S[1,1] coeff 1.
        # But M_S is a 2x2 matrix.
        # The result of the low-level Strassen is the 2x2 matrix M_S.
        # So M_S[0,0] contributes to C block 00 with coeff 1.
        # M_S[0,1] contributes to C block 01 with coeff 0?
        # No, M_S is the *value* of the product.
        # The formula is C_block = sum S coeff[S, block] * M_S.
        # So for a given S, M_S contributes to C_block with coeff coeff[S, block].
        # And M_S is a 2x2 matrix.
        # So M_S[i,j] contributes to C_block[i,j] (which is a 2x2 block of C).
        
        # So, for a given S:
        # M_S[0,0] contributes to C[0:2, 0:2] with coeff c_block_coeffs[S][0,0].
        # M_S[0,1] contributes to C[0:2, 2:4] with coeff c_block_coeffs[S][0,1].
        # M_S[1,0] contributes to C[2:4, 0:2] with coeff c_block_coeffs[S][1,0].
        # M_S[1,1] contributes to C[2:4, 2:4] with coeff c_block_coeffs[S][1,1].
        
        # Let's re-verify the high-level coefficients.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # S=1 (M1): C00:1, C11:1. Others 0.
        # S=2 (M2): C10:1, C11:-1.
        # S=3 (M3): C01:1, C11:1.
        # S=4 (M4): C00:1, C10:1.
        # S=5 (M5): C00:-1, C01:1.
        # S=6 (M6): C11:1.
        # S=7 (M7): C00:1, C01:1.
        
        [[1, 0], [0, 1]], # S=1
        [[0, 0], [1, -1]], # S=2
        [[0, 1], [0, 1]], # S=3
        [[1, 0], [1, 0]], # S=4
        [[-1, 1], [0, 0]], # S=5
        [[0, 0], [0, 1]], # S=6
        [[1, 1], [0, 0]]  # S=7
    ]
    
    for S in range(7):
        for t in range(7):
            u_t = u_strassen[t]
            v_t = v_strassen[t]
            w_t = w_strassen[t]
            
            U_vec = np.zeros(16)
            for li in range(4):
                coeff = u_t[li]
                if coeff == 0: continue
                for block_idx, b_coeff in a_coeffs[S][li]:
                    I, J = divmod(block_idx, 2)
                    for i_l in range(2):
                        for j_l in range(2):
                            i = 2*I + i_l
                            j = 2*J + j_l
                            idx = i*4 + j
                            U_vec[idx] += coeff * b_coeff
            
            V_vec = np.zeros(20)
            for li in range(4):
                coeff = v_t[li]
                if coeff == 0: continue
                for block_idx, b_coeff in b_coeffs[S][li]:
                    I, J = divmod(block_idx, 2)
                    for i_l in range(2):
                        for j_l in range(2):
                            j = 2*I + i_l
                            k = 2*J + j_l
                            idx = j*5 + k
                            V_vec[idx] += coeff * b_coeff
            
            W_vec = np.zeros(20)
            for li in range(4):
                coeff = w_t[li]
                if coeff == 0: continue
                # li corresponds to M_S[li//2, li%2]
                # M_S is 2x2.
                # M_S[0,0] -> li=0
                # M_S[0,1] -> li=1
                # M_S[1,0] -> li=2
                # M_S[1,1] -> li=3
                
                block_I = li // 2
                block_J = li % 2
                
                # Coefficient from high-level Strassen
                h_coeff = c_block_coeffs[S][block_I][block_J]
                if h_coeff == 0: continue
                
                # M_S[li] is a 2x2 block of C.
                # It contributes to C[2*block_I : 2*block_I+2, 2*block_J : 2*block_J+2]
                for i_l in range(2):
                    for j_l in range(2):
                        i = 2*block_I + i_l
                        k = 2*block_J + j_l
                        idx = k*4 + i # W[k*n+i]
                        W_vec[idx] += coeff * h_coeff
            
            U[:, col_offset] = U_vec
            V[:, col_offset] = V_vec
            W[:, col_offset] = W_vec
            col_offset += 1
            
    # Part 2: Schoolbook for 4x4 x 4x1
    # C[:, 4] = A @ B[:, 4]
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col_offset] = 1
            V[j*5 + 4, col_offset] = 1
            W[4*4 + i, col_offset] = 1 # k=4, i
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
