import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    # Rank 65 decomposition: 49 for 4x4x4 (Strassen) + 16 for 4x4x1 (Schoolbook)
    rank = 65
    
    # U: (n*m, rank) = (16, 65)
    # V: (m*p, rank) = (20, 65)
    # W: (p*n, rank) = (20, 65)
    
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Helper to add a Strassen 2x2x2 term
    # Inputs:
    # u_indices: list of (i, j) for A
    # v_indices: list of (j, k) for B
    # w_indices: list of (k, i) for C
    # coeffs_u, coeffs_v, coeffs_w: coefficients for the linear combinations
    # col: current column index
    
    def add_strassen_2x2(col, u_coeffs, v_coeffs, w_coeffs, u_idx, v_idx, w_idx):
        # u_idx: list of 4 indices into U rows (i*m+j)
        # v_idx: list of 4 indices into V rows (j*p+k)
        # w_idx: list of 4 indices into W rows (k*n+i)
        # u_coeffs: 4 coefficients for U
        # v_coeffs: 4 coefficients for V
        # w_coeffs: 4 coefficients for W
        
        # This adds 1 term to the decomposition? No, Strassen 2x2 uses 7 terms.
        # This function should add ONE scalar multiplication term to the global U, V, W.
        # The "term" corresponds to one scalar multiplication in the Strassen 2x2 algorithm.
        
        # In Strassen 2x2:
        # M1 = (A11+A22)(B11+B22) -> 1 mult
        # M2 = (A21+A22)B11 -> 1 mult
        # M3 = A11(B12-B22) -> 1 mult
        # M4 = A22(B21-B11) -> 1 mult
        # M5 = (A11+A12)B22 -> 1 mult
        # M6 = (A21-A11)(B11+B12) -> 1 mult
        # M7 = (A12-A22)(B21+B22) -> 1 mult
        
        # For each of these 7, we have a specific pattern of coefficients.
        # The indices u_idx, v_idx, w_idx refer to the 4 blocks of the 2x2 matrices.
        # u_idx[0] = A11, u_idx[1] = A12, u_idx[2] = A21, u_idx[3] = A22
        # v_idx[0] = B11, v_idx[1] = B12, v_idx[2] = B21, v_idx[3] = B22
        # w_idx[0] = C11, w_idx[1] = C12, w_idx[2] = C21, w_idx[3] = C22
        
        U[u_idx[0], col] += u_coeffs[0]
        U[u_idx[1], col] += u_coeffs[1]
        U[u_idx[2], col] += u_coeffs[2]
        U[u_idx[3], col] += u_coeffs[3]
        
        V[v_idx[0], col] += v_coeffs[0]
        V[v_idx[1], col] += v_coeffs[1]
        V[v_idx[2], col] += v_coeffs[2]
        V[v_idx[3], col] += v_coeffs[3]
        
        W[w_idx[0], col] += w_coeffs[0]
        W[w_idx[1], col] += w_coeffs[1]
        W[w_idx[2], col] += w_coeffs[2]
        W[w_idx[3], col] += w_coeffs[3]

    # Strassen 2x2 patterns
    # Order: A11, A12, A21, A22
    s_u = [
        [1, 0, 0, 1],  # M1: A11+A22
        [0, 0, 1, 1],  # M2: A21+A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [0, 0, 1, -1], # M6: A21-A11
        [0, 1, 0, -1]  # M7: A12-A22
    ]
    s_v = [
        [1, 0, 0, 1],  # M1: B11+B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12-B22
        [0, 0, 1, -1], # M4: B21-B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11+B12
        [0, 0, 1, 1]   # M7: B21+B22
    ]
    s_w = [
        [1, 0, 0, -1], # M1: C11 = M1 - M5 + M4 - M7 ? 
        # Standard reconstruction:
        # C11 = M1 - M5 + M4 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Let's check.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = A11B11 + A12B21
        # M1 - M5 + M4 - M7
        # = (A11+A22)(B11+B22) - (A11+A12)B22 + A22(B21-B11) - (A12-A22)(B21+B22)
        # = A11B11 + A11B22 + A22B11 + A22B22 - A11B22 - A12B22 + A22B21 - A22B11 - A12B21 - A12B22 + A22B21 + A22B22
        # = A11B11 + A22B21 - A12B21 - 2A12B22 + 2A22B22 + A22B21 ... This is getting messy.
        
        # Let's use the known coefficients for W in Strassen 2x2.
        # C11: M1 - M5 + M4 - M7  -> Coeffs for M1..M7: [1, 0, 0, 1, -1, 0, -1]
        # C12: M3 + M5             -> Coeffs: [0, 0, 1, 0, 1, 0, 0]
        # C21: M2 + M4             -> Coeffs: [0, 0, 0, 1, 0, 0, 0] -- Wait, M2 is index 1, M4 is index 3.
        # C22: M1 - M2 + M3 + M6   -> Coeffs: [1, -1, 1, 0, 0, 1, 0]
        
        # But we are building the tensor decomposition directly.
        # Each scalar multiplication in the 2x2 Strassen corresponds to a rank-1 tensor.
        # The "output" of that rank-1 tensor is a 2x2 matrix with specific coefficients.
        # So for M1, the W coefficients are [1, 0, 0, -1] for C11, C12, C21, C22?
        # No, M1 contributes to C11, C12, C21, C22 with coefficients?
        # M1 = (A11+A22)(B11+B22).
        # In the final sum, M1 is added to C11 and C22, and subtracted from C12 and C21?
        # No, the reconstruction is a linear combination of the M's.
        # C11 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 - 1*M7
        # C12 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
        # C21 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
        # C22 = 1*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
        
        # So for the term corresponding to M1, the W vector is [1, 0, 0, 1] (for C11, C12, C21, C22).
        # For M2, W vector is [0, 0, 1, -1].
        # For M3, W vector is [0, 1, 0, 1].
        # For M4, W vector is [1, 0, 1, 0].
        # For M5, W vector is [-1, 1, 0, 0].
        # For M6, W vector is [0, 0, 0, 1].
        # For M7, W vector is [-1, 0, 0, 0].
    ]
    
    # Correct W coefficients for each of the 7 Strassen products:
    # M1: C11+1, C12+0, C21+0, C22+1 -> [1, 0, 0, 1]
    # M2: C11+0, C12+0, C21+1, C22-1 -> [0, 0, 1, -1]
    # M3: C11+0, C12+1, C21+0, C22+1 -> [0, 1, 0, 1]
    # M4: C11+1, C12+0, C21+1, C22+0 -> [1, 0, 1, 0]
    # M5: C11-1, C12+1, C21+0, C22+0 -> [-1, 1, 0, 0]
    # M6: C11+0, C12+0, C21+0, C22+1 -> [0, 0, 0, 1]
    # M7: C11-1, C12+0, C21+0, C22+0 -> [-1, 0, 0, 0]
    
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0]
    ]

    col = 0
    # 4x4x4 Strassen
    # Blocks of A (4x4):
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in block coords.
    # Block size 2x2.
    # A11 indices: i in 0,1; j in 0,1 -> U rows 0,1,4,5
    # A12 indices: i in 0,1; j in 2,3 -> U rows 2,3,6,7
    # A21 indices: i in 2,3; j in 0,1 -> U rows 8,9,12,13
    # A22 indices: i in 2,3; j in 2,3 -> U rows 10,11,14,15
    
    # Blocks of B (4x4 part):
    # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # B11 indices: j in 0,1; k in 0,1 -> V rows 0,1,4,5
    # B12 indices: j in 0,1; k in 2,3 -> V rows 2,3,6,7
    # B21 indices: j in 2,3; k in 0,1 -> V rows 8,9,12,13
    # B22 indices: j in 2,3; k in 2,3 -> V rows 10,11,14,15
    
    # Blocks of C (4x4 part):
    # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    # C11 indices: k in 0,1; i in 0,1 -> W rows 0,1,4,5 (k*n+i: 0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # C12 indices: k in 2,3; i in 0,1 -> W rows 8,9,12,13
    # C21 indices: k in 0,1; i in 2,3 -> W rows 2,3,6,7
    # C22 indices: k in 2,3; i in 2,3 -> W rows 10,11,14,15
    
    # We need to apply Strassen 2x2 for the 4x4x4 part.
    # This involves 7 "big" Strassen products M1..M7.
    # Each M is a 2x2 matrix product.
    # Each 2x2 matrix product uses 7 scalar products.
    # Total 49 scalar products.
    
    # For each of the 7 big products, we define the U, V, W indices and coefficients.
    # Then for each of those, we apply the 7 Strassen 2x2 patterns.
    
    # Big Product 1: (A11+A22)(B11+B22)
    # U indices: A11 + A22
    # A11 U rows: [0,1,4,5], coeffs [1,1,1,1]
    # A22 U rows: [10,11,14,15], coeffs [1,1,1,1]
    # Combined U for M1: 
    # We need to map the 4x4 blocks to the 2x2 Strassen structure.
    # The "A" in the 2x2 Strassen for M1 is A11+A22.
    # So A11 (of 2x2) = A11 (of 4x4 block) + A22 (of 4x4 block)? No.
    # A11 (2x2) is the top-left 2x2 of the 4x4 matrix A.
    # Wait, A11 (4x4 block) IS the top-left 2x2 of A.
    # So for M1, the input "A" is A11(4x4) + A22(4x4).
    # This is a 4x4 matrix. But we are treating the 4x4 product as 7 products of 2x2 matrices.
    # So M1 is a 2x2 matrix product: (A11+A22) * (B11+B22).
    # Here A11+A22 is a 2x2 matrix? No, A11 is 2x2. A11+A22 is 2x2.
    # Yes, the blocks are 2x2.
    # So M1 = (A11+A22) * (B11+B22) is a 2x2 matrix product.
    # We compute this using Strassen 2x2.
    
    # U indices for A11+A22:
    # A11: rows 0,1,4,5. Coeffs 1.
    # A22: rows 10,11,14,15. Coeffs 1.
    # So U for M1 input: rows [0,1,4,5,10,11,14,15] with coeffs [1,1,1,1,1,1,1,1].
    # But Strassen 2x2 expects 4 "blocks" (which are scalars in the 2x2 case).
    # Here the "blocks" of the 4x4 matrix are the 2x2 submatrices.
    # So for M1, the "A11" (of the 2x2 Strassen) is the 2x2 matrix A11+A22?
    # No. The 4x4 matrix is partitioned into four 2x2 blocks.
    # Top-Left: A11. Top-Right: A12. Bottom-Left: A21. Bottom-Right: A22.
    # M1 = (A11+A22)(B11+B22).
    # The matrix (A11+A22) has:
    # Top-Left 2x2: A11.
    # Top-Right 2x2: A12? No, A11+A22 only has A11 and A22.
    # Wait. A11 is the top-left 2x2 block of A. A22 is the bottom-right 2x2 block.
    # So A11+A22 is a 4x4 matrix where:
    # Top-Left 2x2 is A11.
    # Bottom-Right 2x2 is A22.
    # Top-Right and Bottom-Left are 0.
    # So when we apply Strassen 2x2 to this 4x4 matrix, we treat it as:
    # A'11 = A11 (2x2)
    # A'12 = 0
    # A'21 = 0
    # A'22 = A22 (2x2)
    
    # This means for M1, the U coefficients for the 4 blocks of the 2x2 Strassen are:
    # Block 1 (A'11): A11 -> rows 0,1,4,5 coeffs 1
    # Block 2 (A'12): 0
    # Block 3 (A'21): 0
    # Block 4 (A'22): A22 -> rows 10,11,14,15 coeffs 1
    
    # Similarly for V (B11+B22):
    # Block 1 (B'11): B11 -> rows 0,1,4,5 coeffs 1
    # Block 2 (B'12): 0
    # Block 3 (B'21): 0
    # Block 4 (B'22): B22 -> rows 10,11,14,15 coeffs 1
    
    # And for W (Output C11, C12, C21, C22 of the 2x2 result):
    # The result of M1 is a 2x2 matrix.
    # Its top-left 2x2 goes to C11.
    # Its top-right 2x2 goes to C12.
    # Its bottom-left 2x2 goes to C21.
    # Its bottom-right 2x2 goes to C22.
    # So W indices for the 4 blocks of the 2x2 Strassen output are:
    # Block 1 (C'11): C11 -> rows 0,1,4,5
    # Block 2 (C'12): C12 -> rows 8,9,12,13
    # Block 3 (C'21): C21 -> rows 2,3,6,7
    # Block 4 (C'22): C22 -> rows 10,11,14,15
    
    # We can automate this.
    
    def get_block_rows(block_type, block_id):
        # block_type: 'A', 'B', 'C'
        # block_id: 0 (11), 1 (12), 2 (21), 3 (22)
        if block_type == 'A':
            # i in 0,1 or 2,3; j in 0,1 or 2,3
            i_start = 0 if block_id in [0, 1] else 2
            j_start = 0 if block_id in [0, 2] else 2
            rows = []
            for i in range(i_start, i_start+2):
                for j in range(j_start, j_start+2):
                    rows.append(i*4 + j)
            return rows
        elif block_type == 'B':
            # j in 0,1 or 2,3; k in 0,1 or 2,3
            j_start = 0 if block_id in [0, 1] else 2
            k_start = 0 if block_id in [0, 2] else 2
            rows = []
            for j in range(j_start, j_start+2):
                for k in range(k_start, k_start+2):
                    rows.append(j*5 + k)
            return rows
        elif block_type == 'C':
            # k in 0,1 or 2,3; i in 0,1 or 2,3
            k_start = 0 if block_id in [0, 2] else 2
            i_start = 0 if block_id in [0, 1] else 2
            rows = []
            for k in range(k_start, k_start+2):
                for i in range(i_start, i_start+2):
                    rows.append(k*4 + i)
            return rows

    # Define the 7 big Strassen products
    # Each is a tuple of (U_blocks, V_blocks, W_blocks)
    # U_blocks: dict mapping block_id (0..3) to (coeff, block_id_in_4x4)
    # Actually, it's simpler to just list the non-zero blocks.
    
    big_products = [
        # M1: (A11+A22)(B11+B22)
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        # M2: (A21+A22)B11
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}}, # C21+1, C22-1
        # M3: A11(B12-B22)
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}}, # C12+1, C22+1
        # M4: A22(B21-B11)
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}}, # C11+1, C21+1
        # M5: (A11+A12)B22
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}}, # C11-1, C12+1
        # M6: (A21-A11)(B11+B12)
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}}, # C22+1
        # M7: (A12-A22)(B21+B22)
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: -1}} # C11-1
    ]
    
    for prod in big_products:
        # For each of the 7 Strassen 2x2 terms
        for t in range(7):
            # Get U, V, W coeffs for this term
            u_c = s_u[t]
            v_c = s_v[t]
            w_c = s_w[t]
            
            # Map to actual rows
            # u_c[0] applies to U block 0, u_c[1] to block 1, etc.
            for b in range(4):
                if u_c[b] != 0:
                    # Find which 4x4 block this corresponds to
                    # u_c[b] is the coefficient for the b-th block of the 2x2 matrix
                    # We need to know which 4x4 block is being used.
                    # This is defined in prod['U'].
                    # If b is not in prod['U'], coeff is 0.
                    pass
            
            # This logic is getting complicated. Let's just iterate over the non-zero blocks in prod.
            # For a given Strassen term t, the contribution to U is sum over b of u_c[b] * U_block_b.
            # U_block_b is non-zero only if b is in prod['U'].
            
            for b in range(4):
                if u_c[b] == 0: continue
                if b not in prod['U']: continue
                coeff = u_c[b] * prod['U'][b]
                rows = get_block_rows('A', b)
                for r in rows:
                    U[r, col] += coeff
                    
                if v_c[b] != 0:
                    if b in prod['V']:
                        coeff_v = v_c[b] * prod['V'][b]
                        rows_v = get_block_rows('B', b)
                        for r in rows_v:
                            V[r, col] += coeff_v
                            
                if w_c[b] != 0:
                    if b in prod['W']:
                        coeff_w = w_c[b] * prod['W'][b]
                        rows_w = get_block_rows('C', b)
                        for r in rows_w:
                            W[r, col] += coeff_w
                            
            col += 1
            
    # Now handle the 4x4x1 part
    # A (4x4) * v (4x1)
    # 4 dot products of length 4.
    # i in 0..3, j in 0..3.
    # U[i*4+j, col] = 1
    # V[j*5+4, col] = 1  (k=4)
    # W[4*4+i, col] = 1  (k=4, i=i)
    
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1
            W[4*4+i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
