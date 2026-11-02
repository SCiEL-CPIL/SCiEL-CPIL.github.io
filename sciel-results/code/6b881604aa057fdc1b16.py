import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct U, V, W as lists of lists
    # Rank will be 49 (for 4x4x4 part) + 48 (for 4x4x3 part) = 97
    
    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    
    # Base Strassen 2x2x2 tensors (u, v, w)
    # Indices for 2x2 matrix: 0,1,2,3 (row-major)
    base_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    base_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    base_w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, -C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Strassen 4x4x4 combinations
    # We need to map the 2x2 base tensors to the 4x4 context.
    # For each of the 7 super-terms k, we have a 2x2 matrix X_k and Y_k.
    # X_k and Y_k are linear combinations of the 4 blocks of A and B.
    # Let's define the coefficients for X_k and Y_k blocks.
    
    # Blocks of A (4x4):
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # Each block is 2x2.
    # We represent the linear form for each entry of X_k (which is 2x2) 
    # as a vector of length 16 (coeffs for A's 16 entries).
    
    # Define the 7 X_k and Y_k linear forms.
    # X_k is 2x2. Let's store X_k_coeffs[s, r, c] where s is super-term, r,c in 0,1.
    # Each X_k_coeffs[s, r, c] is a vector of length 16.
    
    X_coeffs = np.zeros((7, 2, 2, 16))
    Y_coeffs = np.zeros((7, 2, 2, 16))
    
    # Helper to set block coefficients
    # Block (br, bc) of A corresponds to indices in 16-vector:
    # A is 4x4. Block (0,0) is rows 0-1, cols 0-1.
    # Index in 16-vector: (br*2 + r) * 4 + (bc*2 + c)
    
    def set_block(coeffs, block_r, block_c, val):
        for r in range(2):
            for c in range(2):
                idx = (block_r * 2 + r) * 4 + (block_c * 2 + c)
                coeffs[idx] = val

    # M1: (A11 + A22) (B11 + B22)
    set_block(X_coeffs[0, 0, 0], 0, 0, 1)
    set_block(X_coeffs[0, 0, 0], 1, 1, 1)
    set_block(X_coeffs[0, 1, 1], 0, 0, 1)
    set_block(X_coeffs[0, 1, 1], 1, 1, 1)
    # Wait, X_k is a 2x2 matrix.
    # X_1 = A11 + A22. This is a 2x2 matrix.
    # So X_1[0,0] is the top-left element of (A11+A22).
    # A11 is 2x2. A22 is 2x2.
    # So X_1[0,0] = A11[0,0] + A22[0,0].
    # My set_block function sets the whole block.
    # I need to set the coefficients for each entry of the 2x2 X_k matrix.
    
    # Let's restart the coefficient definition more carefully.
    # X_k is 2x2.
    # For M1: X_1 = A11 + A22.
    # X_1[0,0] = A[0,0] + A[2,2]
    # X_1[0,1] = A[0,1] + A[2,3]
    # X_1[1,0] = A[1,0] + A[3,2]
    # X_1[1,1] = A[1,1] + A[3,3]
    
    # I will create a function to get the coefficient vector for a specific entry of the block-matrix.
    
    def get_coeff_vec_for_block_entry(block_r, block_c, r, c, val=1):
        vec = np.zeros(16)
        idx = (block_r * 2 + r) * 4 + (block_c * 2 + c)
        vec[idx] = val
        return vec

    # Re-define X_coeffs and Y_coeffs
    # X_coeffs[k, r, c] is a vector of length 16.
    
    # M1
    for r in range(2):
        for c in range(2):
            X_coeffs[0, r, c] = get_coeff_vec_for_block_entry(0, 0, r, c) + get_coeff_vec_for_block_entry(1, 1, r, c)
            Y_coeffs[0, r, c] = get_coeff_vec_for_block_entry(0, 0, r, c) + get_coeff_vec_for_block_entry(1, 1, r, c)
            
    # M2: (A21 + A22) B11
    for r in range(2):
        for c in range(2):
            X_coeffs[1, r, c] = get_coeff_vec_for_block_entry(1, 0, r, c) + get_coeff_vec_for_block_entry(1, 1, r, c)
            Y_coeffs[1, r, c] = get_coeff_vec_for_block_entry(0, 0, r, c)
            
    # M3: A11 (B12 - B22)
    for r in range(2):
        for c in range(2):
            X_coeffs[2, r, c] = get_coeff_vec_for_block_entry(0, 0, r, c)
            Y_coeffs[2, r, c] = get_coeff_vec_for_block_entry(0, 1, r, c) - get_coeff_vec_for_block_entry(1, 1, r, c)
            
    # M4: A22 (B21 - B11)
    for r in range(2):
        for c in range(2):
            X_coeffs[3, r, c] = get_coeff_vec_for_block_entry(1, 1, r, c)
            Y_coeffs[3, r, c] = get_coeff_vec_for_block_entry(1, 0, r, c) - get_coeff_vec_for_block_entry(0, 0, r, c)
            
    # M5: (A11 + A12) B22
    for r in range(2):
        for c in range(2):
            X_coeffs[4, r, c] = get_coeff_vec_for_block_entry(0, 0, r, c) + get_coeff_vec_for_block_entry(0, 1, r, c)
            Y_coeffs[4, r, c] = get_coeff_vec_for_block_entry(1, 1, r, c)
            
    # M6: (A21 - A11) (B11 + B12)
    for r in range(2):
        for c in range(2):
            X_coeffs[5, r, c] = get_coeff_vec_for_block_entry(1, 0, r, c) - get_coeff_vec_for_block_entry(0, 0, r, c)
            Y_coeffs[5, r, c] = get_coeff_vec_for_block_entry(0, 0, r, c) + get_coeff_vec_for_block_entry(0, 1, r, c)
            
    # M7: (A12 - A22) (B21 + B22)
    for r in range(2):
        for c in range(2):
            X_coeffs[6, r, c] = get_coeff_vec_for_block_entry(0, 1, r, c) - get_coeff_vec_for_block_entry(1, 1, r, c)
            Y_coeffs[6, r, c] = get_coeff_vec_for_block_entry(1, 0, r, c) + get_coeff_vec_for_block_entry(1, 1, r, c)

    # Now, construct the 49 rank-1 tensors for the 4x4x4 part.
    # For each super-term k (0-6) and sub-term s (0-6):
    # u = sum over r,c of base_u[s][r*2+c] * X_coeffs[k, r, c]
    # v = sum over r,c of base_v[s][r*2+c] * Y_coeffs[k, r, c]
    # w = sum over r,c of base_w[s][r*2+c] * W_pos_coeffs[k, r, c]
    
    # W_pos_coeffs: Where does M_k contribute in C?
    # C is 4x4. M_k is 2x2.
    # M_k contributes to one of the 4 blocks of C.
    # C11 (0,0) gets M1, M4, M5, M7
    # C12 (0,1) gets M3, M5
    # C21 (1,0) gets M2, M4
    # C22 (1,1) gets M1, M2, M3, M6
    
    # Let's define the mapping from super-term k to C-block and sign.
    # k=0 (M1): C11 (+1), C22 (+1)
    # k=1 (M2): C21 (+1), C22 (-1)
    # k=2 (M3): C12 (+1), C22 (+1)
    # k=3 (M4): C11 (+1), C21 (+1)
    # k=4 (M5): C11 (-1), C12 (+1)
    # k=5 (M6): C22 (+1)
    # k=6 (M7): C11 (+1)
    
    # W_coeffs[k, r, c] is a vector of length 16 (for C 4x4).
    W_coeffs = np.zeros((7, 2, 2, 16))
    
    def set_c_block(coeffs, block_r, block_c, r, c, val):
        idx = (block_r * 2 + r) * 4 + (block_c * 2 + c)
        coeffs[idx] = val

    # M1 (k=0)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[0, r, c], 0, 0, r, c, 1)
            set_c_block(W_coeffs[0, r, c], 1, 1, r, c, 1)
            
    # M2 (k=1)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[1, r, c], 1, 0, r, c, 1)
            set_c_block(W_coeffs[1, r, c], 1, 1, r, c, -1)
            
    # M3 (k=2)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[2, r, c], 0, 1, r, c, 1)
            set_c_block(W_coeffs[2, r, c], 1, 1, r, c, 1)
            
    # M4 (k=3)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[3, r, c], 0, 0, r, c, 1)
            set_c_block(W_coeffs[3, r, c], 1, 0, r, c, 1)
            
    # M5 (k=4)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[4, r, c], 0, 0, r, c, -1)
            set_c_block(W_coeffs[4, r, c], 0, 1, r, c, 1)
            
    # M6 (k=5)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[5, r, c], 1, 1, r, c, 1)
            
    # M7 (k=6)
    for r in range(2):
        for c in range(2):
            set_c_block(W_coeffs[6, r, c], 0, 0, r, c, 1)

    U_part1 = []
    V_part1 = []
    W_part1 = []
    
    for k in range(7):
        for s in range(7):
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            for r in range(2):
                for c in range(2):
                    idx_2x2 = r * 2 + c
                    u_vec += base_u[s][idx_2x2] * X_coeffs[k, r, c]
                    v_vec += base_v[s][idx_2x2] * Y_coeffs[k, r, c]
                    w_vec += base_w[s][idx_2x2] * W_coeffs[k, r, c]
                    
            U_part1.append(u_vec.tolist())
            V_part1.append(v_vec.tolist())
            W_part1.append(w_vec.tolist())
            
    # --- Part 2: 4x4x3 Naive Decomposition (Rank 48) ---
    # A is 4x4, B_right is 4x3.
    # U shape (16, 48), V shape (12, 48), W shape (12, 48)
    # Note: V for this part only involves the last 3 columns of B.
    # B is 4x7. Rows 0-3, Cols 0-6.
    # B_right is Cols 4-6.
    # V index: j*7 + k.
    # For B_right, k is 4,5,6.
    
    U_part2 = []
    V_part2 = []
    W_part2 = []
    
    for i in range(4):
        for j in range(4):
            for k in range(3): # k=0,1,2 in local, 4,5,6 in global
                u = [0] * 16
                v = [0] * 28
                w = [0] * 28
                
                u_idx = i * 4 + j
                u[u_idx] = 1
                
                v_idx = j * 7 + (4 + k)
                v[v_idx] = 1
                
                w_idx = (4 + k) * 4 + i
                w[w_idx] = 1
                
                U_part2.append(u)
                V_part2.append(v)
                W_part2.append(w)
                
    # Combine Part 1 and Part 2
    # Part 1 V and W have length 16 (for 4x4 B and 4x4 C).
    # We need to pad them to length 28 (for 4x7 B and 4x7 C).
    
    U_final = U_part1 + U_part2
    V_final = []
    W_final = []
    
    for i in range(len(V_part1)):
        v = V_part1[i] + [0] * 12
        V_final.append(v)
        
    for i in range(len(W_part1)):
        w = W_part1[i] + [0] * 12
        W_final.append(w)
        
    V_final += V_part2
    W_final += W_part2
    
    return {"U": U_final, "V": V_final, "W": W_final}
