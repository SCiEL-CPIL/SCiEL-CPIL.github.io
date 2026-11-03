import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists to avoid numpy overhead in construction
    # but numpy is fine for small sizes.
    
    # Helper to add a rank-1 term
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We don't know rank in advance, so we'll use lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # --- Part 1: M_{4,4,4} using Strassen (Rank 49) ---
    # Strassen 2x2 decomposition
    # A, B are 2x2.
    # u, v, w vectors of length 4.
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen 7 terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # u vectors (coeffs for A)
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 0, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 1, 0, 0]
    u6 = [ -1, 0, 1, 0] # A10 - A00
    u7 = [0, 1, 0, -1] # A01 - A11
    
    # v vectors (coeffs for B)
    v1 = [1, 0, 0, 1]
    v2 = [1, 0, 0, 0]
    v3 = [0, -1, 1, 0] # B10 - B11
    v4 = [ -1, 1, 0, 0] # B01 - B00
    v5 = [0, 0, 0, 1]
    v6 = [1, 1, 0, 0]
    v7 = [0, 0, 1, 1]
    
    # w vectors (coeffs for C)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # w1 contributes to C00 (1), C11 (1)
    w1 = [1, 0, 0, 1]
    # w2 contributes to C10 (1), C11 (-1)
    w2 = [0, 0, 1, -1]
    # w3 contributes to C01 (1), C11 (1)
    w3 = [0, 1, 0, 1]
    # w4 contributes to C00 (1), C10 (1)
    w4 = [1, 0, 1, 0]
    # w5 contributes to C00 (-1), C01 (1)
    w5 = [-1, 1, 0, 0]
    # w6 contributes to C11 (1)
    w6 = [0, 0, 0, 1]
    # w7 contributes to C00 (1)
    w7 = [1, 0, 0, 0]
    
    s_u = [u1, u2, u3, u4, u5, u6, u7]
    s_v = [v1, v2, v3, v4, v5, v6, v7]
    s_w = [w1, w2, w3, w4, w5, w6, w7]
    
    # Strassen 4x4 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Reconstruction:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map A entries (16) to block entries.
    # A indices:
    # 0,1,4,5 -> A11
    # 2,3,6,7 -> A12
    # 8,9,12,13 -> A21
    # 10,11,14,15 -> A22
    
    # B indices:
    # 0,1,4,5 -> B11
    # 2,3,6,7 -> B12
    # 8,9,12,13 -> B21
    # 10,11,14,15 -> B22
    
    # C indices:
    # 0,1,4,5 -> C11
    # 2,3,6,7 -> C12
    # 8,9,12,13 -> C21
    # 10,11,14,15 -> C22
    
    # For each P_b, we have linear combos of A and B.
    # Then we apply the 2x2 Strassen decomposition to the product.
    
    # Define block linear combos
    # Each is a dict: {block_name: coeff}
    # But we need vectors of length 16.
    
    def get_block_vector(blocks, block_indices, length=16):
        vec = [0.0] * length
        for b_name, coeff in blocks.items():
            for idx in block_indices[b_name]:
                vec[idx] += coeff
        return vec
    
    # Block indices in 16-dim space
    b_idx = {
        'A11': [0, 1, 4, 5],
        'A12': [2, 3, 6, 7],
        'A21': [8, 9, 12, 13],
        'A22': [10, 11, 14, 15],
        'B11': [0, 1, 4, 5],
        'B12': [2, 3, 6, 7],
        'B21': [8, 9, 12, 13],
        'B22': [10, 11, 14, 15],
        'C11': [0, 1, 4, 5],
        'C12': [2, 3, 6, 7],
        'C21': [8, 9, 12, 13],
        'C22': [10, 11, 14, 15]
    }
    
    # P_b definitions
    # P_b = L_A_b * L_B_b
    # L_A_b is 2x2 matrix.
    
    # We iterate over the 7 P's.
    # For each P, we define the 2x2 matrix L_A and L_B.
    # Then we iterate over the 7 inner Strassen terms.
    
    P_defs = [
        # 1: (A11+A22)(B11+B22)
        ( {'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1} ),
        # 2: (A21+A22)B11
        ( {'A21': 1, 'A22': 1}, {'B11': 1} ),
        # 3: A11(B21-B22)
        ( {'A11': 1}, {'B21': 1, 'B22': -1} ),
        # 4: A22(B12-B11)
        ( {'A22': 1}, {'B12': 1, 'B11': -1} ),
        # 5: (A11+A12)B22
        ( {'A11': 1, 'A12': 1}, {'B22': 1} ),
        # 6: (A21-A11)(B11+B12)
        ( {'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1} ),
        # 7: (A12-A22)(B21+B22)
        ( {'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1} )
    ]
    
    # Reconstruction coeffs for C blocks from P's
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For each P_b, we need to know how it contributes to each C block.
    # C_block = sum_b alpha_{b, C_block} P_b
    
    # alpha[b][c_block]
    # b is 0..6 (P1..P7)
    # c_block is 'C11', 'C12', 'C21', 'C22'
    
    alpha = [
        # P1
        {'C11': 1, 'C12': 0, 'C21': 0, 'C22': 1},
        # P2
        {'C11': 0, 'C12': 0, 'C21': 1, 'C22': -1},
        # P3
        {'C11': 0, 'C12': 1, 'C21': 0, 'C22': 1},
        # P4
        {'C11': 1, 'C12': 0, 'C21': 1, 'C22': 0},
        # P5
        {'C11': -1, 'C12': 1, 'C21': 0, 'C22': 0},
        # P6
        {'C11': 0, 'C12': 0, 'C21': 0, 'C22': 1},
        # P7
        {'C11': 1, 'C12': 0, 'C21': 0, 'C22': 0}
    ]
    
    # Generate terms for M_{4,4,4}
    for b_idx_p, (la_def, lb_def) in enumerate(P_defs):
        # Construct L_A and L_B as 2x2 matrices?
        # No, we just need the linear combination vectors for A and B.
        # But the inner Strassen works on 2x2 matrices.
        # The "matrix" L_A is formed by combining A blocks.
        # L_A[x,y] = sum_{block} coeff_{block} A_{block}[x,y]
        
        # We need to form the 16-dim vector for U and V.
        # For a specific inner term t (u_t, v_t, w_t):
        # The term is (u_t^T L_A) (v_t^T L_B) w_t^T
        # u_t^T L_A is a linear combination of A entries.
        # L_A[x,y] is a linear combination of A entries.
        # So u_t^T L_A = sum_{x,y} u_t[x,y] L_A[x,y]
        # = sum_{x,y} u_t[x,y] sum_{block} coeff_{block} A_{block}[x,y]
        # = sum_{block} coeff_{block} sum_{x,y} u_t[x,y] A_{block}[x,y]
        
        # So for each block in la_def:
        #   for each x,y in 0..1:
        #       if u_t[x,y] != 0:
        #           add u_t[x,y] * coeff to U vector at A_block[x,y] index
        
        u_t_list = s_u
        v_t_list = s_v
        w_t_list = s_w
        
        for t in range(7):
            u_t = u_t_list[t]
            v_t = v_t_list[t]
            w_t = w_t_list[t]
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            
            # Process L_A
            for b_name, coeff in la_def.items():
                # b_name is like 'A11'
                # We need to map 'A11' to indices in 16-dim A
                # And x,y in 0..1 to local indices in block
                # Block A11: indices 0,1,4,5.
                # x=0,y=0 -> 0; x=0,y=1 -> 1; x=1,y=0 -> 4; x=1,y=1 -> 5.
                # General mapping:
                # Block row r (0,1), col c (0,1).
                # A11: r=0,c=0. A12: r=0,c=1. A21: r=1,c=0. A22: r=1,c=1.
                pass # Logic below
            
            # Helper to get block r,c indices
            def get_block_indices(r, c):
                # 4x4 matrix, 2x2 blocks.
                # Block (r,c) covers rows 2r, 2r+1 and cols 2c, 2c+1.
                idxs = []
                for i in range(2):
                    for j in range(2):
                        row = 2*r + i
                        col = 2*c + j
                        idxs.append(row * 4 + col)
                return idxs
            
            # Map block names to (r,c)
            block_rc = {
                'A11': (0,0), 'A12': (0,1), 'A21': (1,0), 'A22': (1,1),
                'B11': (0,0), 'B12': (0,1), 'B21': (1,0), 'B22': (1,1)
            }
            
            for b_name, coeff in la_def.items():
                r, c = block_rc[b_name]
                idxs = get_block_indices(r, c)
                # u_t is [u00, u01, u10, u11]
                # idxs are [00, 01, 10, 11] of the block
                for k in range(4):
                    val = u_t[k] * coeff
                    if val != 0:
                        U_vec[idxs[k]] += val
                        
            for b_name, coeff in lb_def.items():
                r, c = block_rc[b_name]
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = v_t[k] * coeff
                    if val != 0:
                        V_vec[idxs[k]] += val
            
            # W vector
            W_vec = [0.0] * 16
            # w_t is [w00, w01, w10, w11]
            # w_t contributes to P_b.
            # P_b contributes to C blocks via alpha[b_idx_p].
            # For each C block, we add alpha * w_t[k] to the corresponding C indices.
            
            for c_name, alpha_val in alpha[b_idx_p].items():
                if alpha_val == 0: continue
                r, c = block_rc[c_name] # C11 etc
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = w_t[k] * alpha_val
                    if val != 0:
                        W_vec[idxs[k]] += val
            
            add_term(U_vec, V_vec, W_vec)
            
    # --- Part 2: M_{4,4,4} again for B2 ---
    # Same as above, but B indices are shifted?
    # No, the decomposition is for A (4x4) and B (4x4).
    # In the full problem, B is 4x9.
    # We split B into B1 (cols 0-3) and B2 (cols 4-7) and B3 (col 8).
    # The U vectors are the same.
    # The V vectors are the same, but we need to place them in the correct columns of the full V matrix.
    # Full V shape: (m*p, rank) = (36, rank).
    # B1 uses V indices 0..15.
    # B2 uses V indices 16..31.
    # B3 uses V indices 32..35.
    
    # We already added 49 terms for B1.
    # We need to add 49 terms for B2.
    # And 16 terms for B3.
    
    # To avoid code duplication, we can store the U, V, W vectors from the first pass
    # and reuse them.
    
    # Let's re-structure: Generate the 49 (U, V, W) tuples for M_{4,4,4} once.
    # Then add them for B1 (V offset 0) and B2 (V offset 16).
    
    # I'll clear the lists and regenerate properly.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Generate M_{4,4,4} terms
    m44_terms = []
    for b_idx_p, (la_def, lb_def) in enumerate(P_defs):
        for t in range(7):
            u_t = s_u[t]
            v_t = s_v[t]
            w_t = s_w[t]
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            for b_name, coeff in la_def.items():
                r, c = block_rc[b_name]
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = u_t[k] * coeff
                    if val != 0:
                        U_vec[idxs[k]] += val
                        
            for b_name, coeff in lb_def.items():
                r, c = block_rc[b_name]
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = v_t[k] * coeff
                    if val != 0:
                        V_vec[idxs[k]] += val
            
            for c_name, alpha_val in alpha[b_idx_p].items():
                if alpha_val == 0: continue
                r, c = block_rc[c_name]
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = w_t[k] * alpha_val
                    if val != 0:
                        W_vec[idxs[k]] += val
            
            m44_terms.append((U_vec, V_vec, W_vec))
            
    # Add terms for B1 (cols 0-3 of B)
    for U_vec, V_vec, W_vec in m44_terms:
        # V_vec is for B1 (4x4).
        # Full V is 4x9.
        # V_full indices:
        # B1: 0..15
        # B2: 16..31
        # B3: 32..35
        
        V_full = [0.0] * 36
        for i in range(16):
            V_full[i] = V_vec[i]
            
        add_term(U_vec, V_full, W_vec)
        
    # Add terms for B2 (cols 4-7 of B)
    for U_vec, V_vec, W_vec in m44_terms:
        V_full = [0.0] * 36
        for i in range(16):
            V_full[16+i] = V_vec[i]
            
        add_term(U_vec, V_full, W_vec)
        
    # Add terms for B3 (col 8 of B)
    # M_{4,4,1}
    # C = A v
    # 16 terms.
    for i in range(4):
        for j in range(4):
            U_vec = [0.0] * 16
            U_vec[i*4+j] = 1.0
            
            V_full = [0.0] * 36
            # B3 is col 8.
            # B index for (i, 8) is i*9 + 8?
            # B is 4x9. Row i, Col 8.
            # Index = i * 9 + 8.
            # Wait, V shape is (m*p, rank) = (4*9, rank).
            # V index corresponds to B entry.
            # B is 4x9.
            # Entry (i,j) is at index i*9 + j.
            # So for B3 (col 8), indices are 0*9+8=8, 1*9+8=17, 2*9+8=26, 3*9+8=35.
            
            V_full[i*9+8] = 1.0
            
            W_vec = [0.0] * 16
            # C is 4x1? No, C is 4x9.
            # W shape (p*n, rank) = (9*4, rank).
            # W index corresponds to C entry.
            # C entry (i, 8).
            # Index = i*4 + 8?
            # W[k*n+i, r] convention.
            # k is row of C?
            # Task says: W[k*n+i,r].
            # C is p x n? No, C is n x p.
            # "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C_{ik}.
            # If W is (p*n, rank), and index is k*n+i, then k is row (0..p-1), i is col (0..n-1).
            # So C_{k,i}.
            # My previous assumption: C is n x p.
            # Let's check dimensions.
            # A (n x m), B (m x p). C (n x p).
            # W shape (p*n, rank).
            # If W index is k*n+i, then k in 0..p-1, i in 0..n-1.
            # This implies C is indexed by (k,i) where k is row?
            # If C is n x p, rows are 0..n-1, cols 0..p-1.
            # If W index is k*n+i, and k<p, i<n.
            # Then W_{k*n+i} corresponds to C_{i,k}? Or C_{k,i}?
            # "W[k*n+i,r]"
            # If n=4, p=9.
            # Index 0..35.
            # If k is row of C (0..3) and i is col (0..8), index would be k*9+i.
            # But it says k*n+i.
            # So k is the "slow" index, i is the "fast" index.
            # If k < p and i < n, then k is the row of C?
            # If C is n x p, and we store column-major?
            # Or maybe C is p x n?
            # "multiplying a 4 by 4 matrix by a 4 by 9 matrix"
            # A (4x4), B (4x9). C (4x9).
            # n=4, m=4, p=9.
            # W shape (p*n, rank) = (36, rank).
            # If W[k*n+i], k in 0..8, i in 0..3.
            # Then k is the column of C? i is the row?
            # Let's assume W_{k*n+i} corresponds to C_{i,k}.
            # So C_{row, col} is at W[col*4 + row].
            
            # In my M_{4,4,4} part, I assumed C is 4x4.
            # I used W_vec of length 16.
            # I mapped C blocks to indices 0..15.
            # If C is 4x4, and W index is k*4+i.
            # My get_block_indices for C assumed row-major 4x4?
            # I used `row * 4 + col`.
            # If W index is k*4+i, and k=col, i=row.
            # Then index = col*4 + row.
            # My `get_block_indices` for C used `row * 4 + col`.
            # This is a mismatch if W is column-major.
            
            # Let's re-read: "W[k*n+i,r]"
            # If n=4, index = k*4+i.
            # If C is 4x9, and k is 0..8 (cols), i is 0..3 (rows).
            # Then C_{i,k} is at W[k*4+i].
            
            # In my M_{4,4,4} construction, I treated C as 4x4.
            # I need to map the 4x4 C result to the correct positions in the 4x9 C.
            # B1 corresponds to C cols 0..3.
            # B2 corresponds to C cols 4..7.
            # B3 corresponds to C col 8.
            
            # So for B1, C cols 0..3.
            # C_{i, j} where j in 0..3.
            # W index = j*4 + i.
            
            # My W_vec for M_{4,4,4} was built assuming C is 4x4 with indices 0..15.
            # I used `get_block_indices` which returned `row * 4 + col`.
            # This is row-major.
            # But W is `col * 4 + row` (column-major).
            # So I need to transpose my W_vec logic or the mapping.
            
            # Let's fix the W mapping.
            # In the loop for M_{4,4,4}, I should have mapped C_{r,c} to index c*4+r.
            
            # I will regenerate the M_{4,4,4} terms with correct W mapping.
            
    # Reset
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Corrected M_{4,4,4} generation
    m44_terms = []
    for b_idx_p, (la_def, lb_def) in enumerate(P_defs):
        for t in range(7):
            u_t = s_u[t]
            v_t = s_v[t]
            w_t = s_w[t]
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16 # Local 4x4 C
            
            # U, V same as before
            
            for b_name, coeff in la_def.items():
                r, c = block_rc[b_name]
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = u_t[k] * coeff
                    if val != 0:
                        U_vec[idxs[k]] += val
                        
            for b_name, coeff in lb_def.items():
                r, c = block_rc[b_name]
                idxs = get_block_indices(r, c)
                for k in range(4):
                    val = v_t[k] * coeff
                    if val != 0:
                        V_vec[idxs[k]] += val
            
            # W mapping: C_{r,c} -> index c*4 + r
            for c_name, alpha_val in alpha[b_idx_p].items():
                if alpha_val == 0: continue
                r, c = block_rc[c_name]
                # Get local indices in 4x4 C block
                # Block (r,c) covers rows 2r, 2r+1 and cols 2c, 2c+1
                # Local x,y in 0..1
                for x in range(2):
                    for y in range(2):
                        row = 2*r + x
                        col = 2*c + y
                        # w_t index for local (x,y) is x*2 + y?
                        # w_t = [w00, w01, w10, w11]
                        # x=0,y=0 -> 0; x=0,y=1 -> 1; x=1,y=0 -> 2; x=1,y=1 -> 3.
                        w_idx = x*2 + y
                        val = w_t[w_idx] * alpha_val
                        if val != 0:
                            # C index in 4x4 space (column-major)
                            c_idx = col * 4 + row
                            W_vec[c_idx] += val
            
            m44_terms.append((U_vec, V_vec, W_vec))
            
    # Add B1 (cols 0-3)
    for U_vec, V_vec, W_vec in m44_terms:
        V_full = [0.0] * 36
        for i in range(16):
            V_full[i] = V_vec[i]
            
        W_full = [0.0] * 36
        # W_vec is for C cols 0..3.
        # C_{row, col} where col in 0..3.
        # W index = col*4 + row.
        # My W_vec index is col*4 + row.
        # So W_full[i] = W_vec[i] for i in 0..15.
        for i in range(16):
            W_full[i] = W_vec[i]
            
        add_term(U_vec, V_full, W_full)
        
    # Add B2 (cols 4-7)
    for U_vec, V_vec, W_vec in m44_terms:
        V_full = [0.0] * 36
        for i in range(16):
            V_full[16+i] = V_vec[i]
            
        W_full = [0.0] * 36
        # C cols 4..7.
        # C_{row, col} where col in 4..7.
        # W index = col*4 + row.
        # col = 4 + c_local.
        # index = (4 + c_local)*4 + row = 16 + c_local*4 + row.
        # W_vec index is c_local*4 + row.
        # So W_full[16+i] = W_vec[i].
        for i in range(16):
            W_full[16+i] = W_vec[i]
            
        add_term(U_vec, V_full, W_full)
        
    # Add B3 (col 8)
    for i in range(4):
        for j in range(4):
            U_vec = [0.0] * 16
            U_vec[i*4+j] = 1.0
            
            V_full = [0.0] * 36
            # B col 8, row j.
            # V index = j*9 + 8.
            V_full[j*9+8] = 1.0
            
            W_full = [0.0] * 36
            # C col 8, row i.
            # W index = 8*4 + i = 32 + i.
            W_full[32+i] = 1.0
            
            add_term(U_vec, V_full, W_full)
            
    # Convert to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T # (16, rank)
    V = np.array(V_cols).T # (36, rank)
    W = np.array(W_cols).T # (36, rank)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
