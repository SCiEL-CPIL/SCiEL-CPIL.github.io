import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        nonlocal rank
        U.append([0] * 0) # Placeholder, we build lists of lists
        # Actually, let's build the matrices as lists of lists directly.
        # Since we don't know the final rank, we append columns.
        pass

    # Let's use numpy arrays for construction and convert at the end.
    # Max rank estimate: 100.
    max_rank = 100
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    col = 0
    
    # --- Part 1: 4x4x4 Strassen ---
    # Strassen coefficients for 2x2 blocks
    # A blocks: 11, 12, 21, 22
    # B blocks: 11, 12, 21, 22
    
    # Outer Strassen steps
    # s=1: A=(11+22), B=(11+22), C_contrib: C11+=1, C22+=1
    # s=2: A=(21+22), B=(11),     C_contrib: C21+=1, C22-=1
    # s=3: A=(11),     B=(12-22), C_contrib: C12+=1, C22+=1
    # s=4: A=(22),     B=(21-11), C_contrib: C11+=1, C21+=1
    # s=5: A=(11+12), B=(22),     C_contrib: C11-=1, C12+=1
    # s=6: A=(21-11), B=(11+12), C_contrib: C22+=1
    # s=7: A=(12-22), B=(21+22), C_contrib: C11+=1
    
    # Inner Strassen (for each 2x2 product)
    # The inner Strassen produces 7 terms.
    # We need to map the inner 2x2 indices to the outer 4x4 indices.
    
    # Let's define the inner Strassen factors for 2x2x2.
    # u_inner[4, 7], v_inner[4, 7], w_inner[4, 7]
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Constructing inner Strassen factors:
    # This is tedious to hardcode. Let's use the known structure.
    # Actually, we can just use the schoolbook for the inner 2x2 if we want simplicity, 
    # but that would be rank 4, total 7*4=28 for 4x4x4? No, 7*4=28 is rank for 4x4x4?
    # No, if we use schoolbook for the inner 2x2, the rank is 7 * 4 = 28.
    # But the rank of 4x4x4 is 49. Why?
    # Because Strassen reduces the number of *matrix* multiplications, but each matrix multiplication still has cost.
    # If we use schoolbook for the base case, the rank is $7 \times 4 = 28$?
    # Wait. The rank of the tensor is the number of scalar multiplications.
    # If we replace each 2x2 multiplication with 4 scalar multiplications, the total is 28.
    # Is the rank of 4x4x4 actually 28?
    # No, the rank of 4x4x4 is at least 49? No, lower bound is 49?
    # Actually, the rank of 4x4x4 is 49.
    # Why? Because the inner 2x2 multiplication has rank 7, not 4.
    # So we MUST use the rank-7 decomposition for the inner 2x2.
    
    # Let's hardcode the rank-7 factors for 2x2x2.
    # Reference: Strassen 1969.
    
    # We will generate them using a helper function that builds the 2x2 Strassen.
    
    def get_2x2_strassen_factors():
        # Returns U, V, W of shape (4, 7)
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # A00, A11 are indices 0, 3? 
        # Let's use standard 2x2 indexing:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # M1: A=(1,0,0,1), B=(1,0,0,1)
        u[:,0] = [1, 0, 0, 1]
        v[:,0] = [1, 0, 0, 1]
        w[:,0] = [1, 0, 0, 1] # C00 += 1, C11 += 1? 
        # Wait, the W factor in the decomposition T[i,j,k] = sum U V W 
        # corresponds to the output C.
        # For 2x2, C has indices (0,0), (0,1), (1,0), (1,1).
        # M1 contributes to C00 and C11.
        # So w[:,0] should be [1, 0, 0, 1].
        
        # M2 = (A10+A11)B00
        u[:,1] = [0, 0, 1, 1]
        v[:,1] = [1, 0, 0, 0]
        w[:,1] = [0, 0, 1, 0] # C10 += 1
        
        # M3 = A00(B01-B11)
        u[:,2] = [1, 0, 0, 0]
        v[:,2] = [0, 1, 0, -1]
        w[:,2] = [0, 1, 0, 0] # C01 += 1
        
        # M4 = A11(B10-B00)
        u[:,3] = [0, 0, 0, 1]
        v[:,3] = [0, 0, 1, -1]
        w[:,3] = [1, 0, 0, 0] # C00 += 1
        
        # M5 = (A00+A01)B11
        u[:,4] = [1, 1, 0, 0]
        v[:,4] = [0, 0, 0, 1]
        w[:,4] = [0, 0, 0, 1] # C11 += 1? 
        # Check C11 = M1 - M2 + M3 + M6?
        # My previous formula: C11 = M1 + M4 - M5 + M7.
        # Let's stick to one consistent set.
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1 -> C00(1), C11(1)
        # M2 -> C10(1), C11(-1)
        # M3 -> C01(1), C11(1)
        # M4 -> C00(1), C10(1)
        # M5 -> C00(-1), C01(1)
        # M6 -> C11(1)
        # M7 -> C00(1)
        
        w[:,0] = [1, 0, 0, 1] # M1
        w[:,1] = [0, 0, 1, -1] # M2
        w[:,2] = [0, 1, 0, 1] # M3
        w[:,3] = [1, 0, 1, 0] # M4
        w[:,4] = [-1, 1, 0, 0] # M5
        w[:,5] = [0, 0, 0, 1] # M6
        w[:,6] = [1, 0, 0, 0] # M7
        
        return u, v, w

    u2, v2, w2 = get_2x2_strassen_factors()
    
    # Now map to 4x4.
    # Outer Strassen steps
    # Each step s has A_coeff (4x4 block matrix of coeffs) and B_coeff.
    # We need to expand these into the 4x4 U and V vectors.
    
    # Helper to add a scaled 2x2 Strassen block to the global matrices
    def add_2x2_block(u_block_coeffs, v_block_coeffs, w_block_coeffs, offset_u, offset_v, offset_w):
        nonlocal col
        for r in range(7):
            # u_block_coeffs is 4x4 (2x2 blocks of 2x2)
            # We add u_block_coeffs * u2[:, r] to U
            # u_block_coeffs maps the 4 inner indices to the 4 outer block indices?
            # No, u_block_coeffs is the linear combination of the 4 2x2 blocks of A.
            # Let A_blocks be a 4x4 matrix (flattened).
            # The contribution to U is:
            # U[:, col] += u_block_coeffs @ u2[:, r] ?
            # u_block_coeffs is 4x4. u2[:, r] is 4x1.
            # This gives a 4x1 vector. But U has 16 entries.
            # We need to map the 4 block indices to the 16 global indices.
            
            pass

    # This mapping is complex. 
    # Alternative: Just use schoolbook for 4x4x4? Rank 64.
    # Total 64 + 32 = 96. No improvement.
    
    # Let's try to implement the mapping correctly.
    # A is 4x4. Blocks:
    # A00: (0,0)-(1,1) -> indices 0,1,4,5
    # A01: (0,2)-(1,3) -> indices 2,3,6,7
    # A10: (2,0)-(3,1) -> indices 8,9,12,13
    # A11: (2,2)-(3,3) -> indices 10,11,14,15
    
    # If we have coefficients c00, c01, c10, c11 for the blocks,
    # The vector u_global (size 16) is:
    # u_global[0:4] = c00 * u2_inner? No.
    # u2_inner is the factor for the *scalar* multiplication inside the block.
    # The block product is sum_{r} u_block[r] * v_block[r] * w_block[r].
    # u_block[r] is a 2x2 matrix.
    # So for a specific r, the contribution to U is:
    # U[i, j, col] = (coeff of A_{i_block, j_block}) * u2[i_in, j_in, r]
    
    # Let's define the block coefficients for each outer step s.
    # s=1: A=(1,0,0,1) for blocks (00, 01, 10, 11)?
    # A = A00 + A11. So c00=1, c11=1, others 0.
    
    outer_steps = [
        # (A_coeffs 4x1, B_coeffs 4x1, W_coeffs 4x1)
        # A_coeffs: [c00, c01, c10, c11]
        # B_coeffs: [c00, c01, c10, c11]
        # W_coeffs: [c00, c01, c10, c11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Block index mapping:
    # 0: (0,0) -> global 0,1,4,5
    # 1: (0,1) -> global 2,3,6,7
    # 2: (1,0) -> global 8,9,12,13
    # 3: (1,1) -> global 10,11,14,15
    
    # Inner index mapping for 2x2:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    for s in range(7):
        a_c = outer_steps[s][0]
        b_c = outer_steps[s][1]
        w_c = outer_steps[s][2]
        
        for r in range(7):
            # u2[r], v2[r], w2[r] are the inner factors
            # We need to expand them into global U, V, W
            
            # U expansion
            u_global = np.zeros(16)
            for block_idx in range(4):
                if a_c[block_idx] != 0:
                    # Add a_c[block_idx] * u2[:, r] to the corresponding 4 entries
                    # Determine global indices for this block
                    # Block 0: 0,1,4,5
                    # Block 1: 2,3,6,7
                    # Block 2: 8,9,12,13
                    # Block 3: 10,11,14,15
                    if block_idx == 0: idxs = [0,1,4,5]
                    elif block_idx == 1: idxs = [2,3,6,7]
                    elif block_idx == 2: idxs = [8,9,12,13]
                    else: idxs = [10,11,14,15]
                    
                    for inner_idx, g_idx in enumerate(idxs):
                        u_global[g_idx] += a_c[block_idx] * u2[inner_idx, r]
                        
            # V expansion
            v_global = np.zeros(24) # m*p = 4*6
            for block_idx in range(4):
                if b_c[block_idx] != 0:
                    # B blocks are columns 0-3.
                    # Block 0: (0,0)-(1,1) of B -> B[0:2, 0:2] -> indices 0,1,6,7?
                    # V index: j*6 + k.
                    # Block 0 (j=0,1; k=0,1):
                    # (0,0)->0, (0,1)->1, (1,0)->6, (1,1)->7
                    if block_idx == 0: idxs = [0,1,6,7]
                    elif block_idx == 1: idxs = [2,3,8,9] # j=0,1; k=2,3
                    elif block_idx == 2: idxs = [12,13,18,19] # j=2,3; k=0,1
                    else: idxs = [14,15,20,21] # j=2,3; k=2,3
                    
                    for inner_idx, g_idx in enumerate(idxs):
                        v_global[g_idx] += b_c[block_idx] * v2[inner_idx, r]
                        
            # W expansion
            w_global = np.zeros(24) # p*n = 6*4
            for block_idx in range(4):
                if w_c[block_idx] != 0:
                    # C blocks are columns 0-3.
                    # W index: k*4 + i.
                    # Block 0 (k=0,1; i=0,1):
                    # (0,0)->0, (1,0)->4, (0,1)->1, (1,1)->5?
                    # k=0, i=0 -> 0
                    # k=1, i=0 -> 4
                    # k=0, i=1 -> 1
                    # k=1, i=1 -> 5
                    if block_idx == 0: idxs = [0,4,1,5]
                    elif block_idx == 1: idxs = [2,6,3,7] # k=0,1; i=2,3
                    elif block_idx == 2: idxs = [8,12,9,13] # k=2,3; i=0,1
                    else: idxs = [10,14,11,15] # k=2,3; i=2,3
                    
                    for inner_idx, g_idx in enumerate(idxs):
                        w_global[g_idx] += w_c[block_idx] * w2[inner_idx, r]
                        
            U[:, col] = u_global
            V[:, col] = v_global
            W[:, col] = w_global
            col += 1
            
    # --- Part 2: 4x4x2 Schoolbook ---
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*4 + j, col] = 1
                V[j*6 + k, col] = 1
                W[k*4 + i, col] = 1
                col += 1
                
    # Trim
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
