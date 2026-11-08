import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    
    # Define blocks as lists of global indices (row, col)
    # A11: rows 0-1, cols 0-1
    A11 = [(0,0), (0,1), (1,0), (1,1)]
    A12 = [(0,2), (0,3), (1,2), (1,3)]
    A21 = [(2,0), (2,1), (3,0), (3,1)]
    A22 = [(2,2), (2,3), (3,2), (3,3)]
    
    B11 = [(0,0), (0,1), (1,0), (1,1)]
    B12 = [(0,2), (0,3), (1,2), (1,3)]
    B21 = [(2,0), (2,1), (3,0), (3,1)]
    B22 = [(2,2), (2,3), (3,2), (3,3)]
    
    C11 = [(0,0), (0,1), (1,0), (1,1)] # Maps to C indices 0,1,4,5
    C12 = [(0,2), (0,3), (1,2), (1,3)] # Maps to C indices 2,3,6,7
    C21 = [(2,0), (2,1), (3,0), (3,1)] # Maps to C indices 8,9,12,13
    C22 = [(2,2), (2,3), (3,2), (3,3)] # Maps to C indices 10,11,14,15
    
    # Outer Strassen Terms
    # Each term is (P_blocks, Q_blocks)
    # P_blocks: list of (block_name, coeff)
    outer_terms = [
        ([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)]), # M1
        ([(A21, 1), (A22, 1)], [(B11, 1)]),           # M2
        ([(A11, 1)], [(B12, 1), (B22, -1)]),          # M3
        ([(A22, 1)], [(B21, 1), (B11, -1)]),          # M4
        ([(A11, 1), (A12, 1)], [(B22, 1)]),           # M5
        ([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)]),# M6
        ([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)]) # M7
    ]
    
    # Inner Strassen Terms
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {local_idx: coeff} for p_00, p_01, p_10, p_11
    # B_coeffs: dict {local_idx: coeff} for q_00, q_01, q_10, q_11
    # C_coeffs: dict {local_idx: coeff} for r_00, r_01, r_10, r_11
    # local_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    inner_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),       # M1
        ({2:1, 3:1}, {0:1}, {2:1}),                  # M2
        ({0:1}, {1:1, 3:-1}, {1:1}),                 # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),            # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),            # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),            # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1})        # M7
    ]
    
    # Helper to get global index from local block index
    def get_global_idx(block, local_idx):
        return block[local_idx]
        
    for P_blocks, Q_blocks in outer_terms:
        # Build P and Q as 2x2 matrices of linear forms
        # P[r][c] = list of (global_idx, coeff)
        P = [[[] for _ in range(2)] for _ in range(2)]
        Q = [[[] for _ in range(2)] for _ in range(2)]
        
        for block, coeff in P_blocks:
            for lr in range(2):
                for lc in range(2):
                    g_idx = get_global_idx(block, lr*2+lc)
                    P[lr][lc].append((g_idx, coeff))
                    
        for block, coeff in Q_blocks:
            for lr in range(2):
                for lc in range(2):
                    g_idx = get_global_idx(block, lr*2+lc)
                    Q[lr][lc].append((g_idx, coeff))
                    
        for A_coeffs, B_coeffs, C_coeffs in inner_terms:
            u_vec = [0.0] * 16
            v_vec = [0.0] * 20 # m*p = 4*5 = 20
            w_vec = [0.0] * 20 # p*n = 5*4 = 20
            
            # Construct U vector
            for local_idx, coeff in A_coeffs.items():
                lr = local_idx // 2
                lc = local_idx % 2
                for g_idx, block_coeff in P[lr][lc]:
                    u_vec[g_idx[0]*4 + g_idx[1]] += coeff * block_coeff
                    
            # Construct V vector
            for local_idx, coeff in B_coeffs.items():
                lr = local_idx // 2
                lc = local_idx % 2
                for g_idx, block_coeff in Q[lr][lc]:
                    v_vec[g_idx[0]*4 + g_idx[1]] += coeff * block_coeff
                    
            # Construct W vector
            # C is 4x4 for this part.
            # r_00 -> C[0,0] (idx 0)
            # r_01 -> C[0,1] (idx 1)
            # r_10 -> C[1,0] (idx 4)
            # r_11 -> C[1,1] (idx 5)
            # Note: The C_coeffs are for the 2x2 result block.
            # We need to map the 2x2 result block to the global C matrix.
            # The outer term corresponds to one of the 4 blocks of C.
            # Wait, the outer Strassen produces C11, C12, C21, C22.
            # Which block does this outer term contribute to?
            # Actually, the outer Strassen terms M1..M7 are used to compute ALL blocks of C.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So, for a given outer term (say M1), it contributes to multiple C blocks.
            # The "C_coeffs" in inner_terms are for the 2x2 matrix R = P Q.
            # This R is NOT the final C block.
            # The final C blocks are linear combinations of the 7 outer terms.
            
            # My current loop structure is iterating over Outer Terms and Inner Terms.
            # This generates 49 terms.
            # Each term is a scalar multiplication.
            # The W-vector for this term must reflect how this specific scalar product contributes to the final C.
            
            # Let's re-evaluate.
            # The tensor is T = sum (u_i A v_i^T) (w_i B z_i^T) ...
            # The decomposition is C = sum_r u_r (A u_r) (v_r^T B) v_r? No.
            # C_{ij} = sum_r U_{i,r} V_{j,r} W_{k,r} ...
            
            # In my construction:
            # Term r corresponds to one inner Strassen multiplication.
            # u_r is the linear form in A.
            # v_r is the linear form in B.
            # w_r is the linear form in C.
            
            # For a specific Outer Term M_k and Inner Term m_l:
            # The product is L_A * L_B.
            # This product contributes to the entries of the 2x2 matrix R_k (the k-th outer term).
            # The entries of R_k are linear combinations of the final C entries.
            
            # Example: Outer Term M1.
            # R1 = (A11+A22)(B11+B22).
            # R1 is a 2x2 matrix.
            # C11 = R1[0,0] + ...
            # C12 = ...
            # Actually, the Strassen algorithm defines:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Here M1, M2... are 2x2 matrices.
            # So C11[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
            # C11[0,1] = M1[0,1] + M4[0,1] - M5[0,1] + M7[0,1]
            # etc.
            
            # So, for a specific Inner Term within Outer Term M1:
            # It produces a value for R1[r,c] (where r,c in 0..1).
            # This value contributes to C11[r,c] with coeff +1.
            # It does NOT contribute to C12, C21, C22 directly from M1.
            # Wait, M1 contributes to C11 and C22.
            # C11 = M1 + ...
            # C22 = M1 - ...
            # So R1[0,0] contributes to C11[0,0] (+1) and C22[0,0] (+1).
            # R1[0,1] contributes to C11[0,1] (+1) and C22[0,1] (+1).
            # R1[1,0] contributes to C11[1,0] (+1) and C22[1,0] (+1).
            # R1[1,1] contributes to C11[1,1] (+1) and C22[1,1] (+1).
            
            # So I need to know which Outer Term I am in, and apply the correct mapping to C.
            
            # Let's define the mapping from Outer Term Index to C Block Coefficients.
            # outer_idx 0 (M1): C11(+1), C22(+1)
            # outer_idx 1 (M2): C21(+1), C22(-1)
            # outer_idx 2 (M3): C12(+1), C22(+1)
            # outer_idx 3 (M4): C11(+1), C21(+1)
            # outer_idx 4 (M5): C11(-1), C12(+1)
            # outer_idx 5 (M6): C22(+1)
            # outer_idx 6 (M7): C11(+1), C22(+1)
            
            pass # Need to restructure loop to pass outer_idx
            
    # Re-implementing loop with outer_idx
    
    U = []
    V = []
    W = []
    
    outer_mappings = [
        {0: 1, 3: 1},      # M1 -> C11, C22
        {2: 1, 3: -1},     # M2 -> C21, C22
        {1: 1, 3: 1},      # M3 -> C12, C22
        {0: 1, 2: 1},      # M4 -> C11, C21
        {0: -1, 1: 1},     # M5 -> C11, C12
        {3: 1},            # M6 -> C22
        {0: 1, 3: 1}       # M7 -> C11, C22
    ]
    
    # C Block Indices
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    block_indices = [
        [0, 1, 4, 5], # C11
        [2, 3, 6, 7], # C12
        [8, 9, 12, 13], # C21
        [10, 11, 14, 15] # C22
    ]
    
    for outer_idx, (P_blocks, Q_blocks) in enumerate(outer_terms):
        P = [[[] for _ in range(2)] for _ in range(2)]
        Q = [[[] for _ in range(2)] for _ in range(2)]
        
        for block, coeff in P_blocks:
            for lr in range(2):
                for lc in range(2):
                    g_idx = get_global_idx(block, lr*2+lc)
                    P[lr][lc].append((g_idx, coeff))
                    
        for block, coeff in Q_blocks:
            for lr in range(2):
                for lc in range(2):
                    g_idx = get_global_idx(block, lr*2+lc)
                    Q[lr][lc].append((g_idx, coeff))
                    
        for A_coeffs, B_coeffs, C_coeffs in inner_terms:
            u_vec = [0.0] * 16
            v_vec = [0.0] * 20
            w_vec = [0.0] * 20
            
            for local_idx, coeff in A_coeffs.items():
                lr = local_idx // 2
                lc = local_idx % 2
                for g_idx, block_coeff in P[lr][lc]:
                    u_vec[g_idx[0]*4 + g_idx[1]] += coeff * block_coeff
                    
            for local_idx, coeff in B_coeffs.items():
                lr = local_idx // 2
                lc = local_idx % 2
                for g_idx, block_coeff in Q[lr][lc]:
                    v_vec[g_idx[0]*4 + g_idx[1]] += coeff * block_coeff
                    
            for r_local, c_local in [(0,0), (0,1), (1,0), (1,1)]:
                c_coeff = C_coeffs.get(r_local*2+c_local, 0)
                if c_coeff != 0:
                    for c_block_idx, c_block_coeff in outer_mappings[outer_idx].items():
                        # c_block_idx is 0..3 corresponding to C11, C12, C21, C22
                        # r_local, c_local is position within that 2x2 block
                        global_c_idx = block_indices[c_block_idx][r_local*2+c_local]
                        w_vec[global_c_idx] += c_coeff * c_block_coeff
                        
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
    # --- Part 2: 4x4x1 (Rank 16) ---
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 20
            w_vec = [0.0] * 20
            
            u_vec[i*4+k] = 1.0
            v_vec[k*4+4] = 1.0
            w_vec[i*4+4] = 1.0
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
            
    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    
    return {"U": U, "V": V, "W": W}
