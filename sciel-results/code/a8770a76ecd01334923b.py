import numpy as np

def get_strassen_4x4_triples():
    """
    Generates the 49 triples (u, v, w) for 4x4 matrix multiplication using Strassen.
    u, v, w are lists of 16 floats.
    """
    # Strassen 2x2 definition
    # u_2x2: list of 7 vectors of length 4
    # v_2x2: list of 7 vectors of length 4
    # w_2x2: list of 7 vectors of length 4 (coeffs for C_00, C_01, C_10, C_11)
    
    u_2x2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    v_2x2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # w_2x2: recon coefficients for C entries from M1..M7
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M3 + M4
    w_2x2 = [
        [1, 0, 0, 1], # M1 contributes to C00, C11
        [0, 0, 1, 0], # M2 contributes to C10
        [0, 1, 0, 0], # M3 contributes to C01
        [0, 0, 0, 1], # M4 contributes to C11 (and C00 with -1) -> wait, let's just store the full recon matrix
    ]
    # Actually, it's better to store the reconstruction matrix R where R[c, m] is the coeff.
    # C is 4x1 vector (C00, C01, C10, C11). M is 7x1 vector.
    R_2x2 = np.array([
        [1, 0, 0, -1, 0, 1, 0], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 0, 0, 0, 1],  # C10
        [1, 0, -1, 1, 0, 0, 0]  # C11
    ])
    
    # Block level Strassen for 4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the 2x2 Strassen forms to the 4x4 entries.
    # Let's define the 7 block products' U and V forms as 16-dim vectors.
    
    # Helper to get a 16-dim vector from block coefficients
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Entry indices within block: 0,1,2,3 (flattened 2x2)
    
    def make_4x4_vec(block_coeffs, entry_coeffs):
        # block_coeffs: list of 4 coeffs for A00..A11
        # entry_coeffs: list of 4 coeffs for the 4 entries within the block
        # This assumes the form is sum_b block_coeff[b] * (sum_e entry_coeff[e] * A_{b,e})
        # This is only true if the same entry_coeffs apply to all blocks.
        # For Strassen, e.g., A00+A11, the entry coeffs are [1,1,1,1] for all entries?
        # No, A00+A11 means A00[0,0]+A11[0,0], A00[0,1]+A11[0,1], etc.
        # So yes, the entry coeffs are the same for all blocks involved.
        vec = [0.0] * 16
        for b in range(4):
            if block_coeffs[b] != 0:
                for e in range(4):
                    if entry_coeffs[e] != 0:
                        vec[b*4 + e] += block_coeffs[b] * entry_coeffs[e]
        return vec

    # Define the 7 block-level U and V forms
    # U_block[m] is a 16-dim vector for A
    # V_block[m] is a 16-dim vector for B
    
    # M1: (A00+A11) and (B00+B11)
    # U_M1: A00 and A11 have coeff 1. Entry coeffs [1,1,1,1].
    U_M1 = make_4x4_vec([1, 0, 0, 1], [1, 1, 1, 1])
    V_M1 = make_4x4_vec([1, 0, 0, 1], [1, 1, 1, 1])
    
    # M2: (A10+A11) and B00
    # U_M2: A10, A11 coeff 1.
    U_M2 = make_4x4_vec([0, 0, 1, 1], [1, 1, 1, 1])
    # V_M2: B00 coeff 1.
    V_M2 = make_4x4_vec([1, 0, 0, 0], [1, 1, 1, 1])
    
    # M3: A00 and (B01-B11)
    U_M3 = make_4x4_vec([1, 0, 0, 0], [1, 1, 1, 1])
    V_M3 = make_4x4_vec([0, 1, 0, -1], [1, 1, 1, 1])
    
    # M4: A11 and (B10-B00)
    U_M4 = make_4x4_vec([0, 0, 0, 1], [1, 1, 1, 1])
    V_M4 = make_4x4_vec([-1, 0, 1, 0], [1, 1, 1, 1]) # B10 - B00
    
    # M5: (A00+A01) and B11
    U_M5 = make_4x4_vec([1, 1, 0, 0], [1, 1, 1, 1])
    V_M5 = make_4x4_vec([0, 0, 0, 1], [1, 1, 1, 1])
    
    # M6: (A10-A00) and (B00+B01)
    U_M6 = make_4x4_vec([-1, 0, 1, 0], [1, 1, 1, 1])
    V_M6 = make_4x4_vec([1, 1, 0, 0], [1, 1, 1, 1])
    
    # M7: (A01-A11) and (B10+B11)
    U_M7 = make_4x4_vec([0, 1, 0, -1], [1, 1, 1, 1])
    V_M7 = make_4x4_vec([0, 0, 1, 1], [1, 1, 1, 1])
    
    U_blocks = [U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7]
    V_blocks = [V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7]
    
    # Now, for each block product m, we apply the 2x2 Strassen decomposition to the 2x2 blocks.
    # Wait, the "block" is a 2x2 matrix. The Strassen algorithm operates on the 2x2 matrix of blocks.
    # My U_M1 is a linear form on the entire 4x4 matrix.
    # But Strassen for 4x4 says:
    # Compute 7 products of 2x2 matrices.
    # Each 2x2 matrix product can be done with 7 scalar products.
    # So I need to decompose each U_Mm and V_Mm further?
    # No. U_Mm is a linear form on A. It produces a scalar? No.
    # In the 4x4 Strassen, M1 is a 2x2 matrix.
    # M1 = (A00+A11)(B00+B11).
    # The entries of M1 are 2x2 matrix products? No, M1 is a 2x2 matrix where each entry is a scalar product of 2x2 matrices?
    # No, A00 is a 2x2 matrix. A00+A11 is a 2x2 matrix.
    # The product (A00+A11)(B00+B11) is a 2x2 matrix multiplication.
    # So M1 is a 2x2 matrix.
    # To compute this 2x2 matrix product, we use the 2x2 Strassen algorithm.
    # This requires 7 scalar products.
    # So for each of the 7 block products, we generate 7 scalar product triples.
    # Total 49.
    
    # How to generate the triples?
    # For block product m:
    # We have a 2x2 matrix L_m (from A) and R_m (from B).
    # L_m is a 2x2 matrix of 2x2 matrices? No.
    # L_m is a 2x2 matrix where each entry is a 2x2 matrix.
    # Actually, L_m is just a 2x2 matrix of linear forms on the 4 entries of the blocks?
    # No, L_m[0,0] is the (0,0) entry of the 2x2 matrix A00+A11.
    # L_m[0,0] = A00[0,0] + A11[0,0].
    # This is a linear form on the 16 entries of A.
    
    # So for block product m, we have 4 linear forms for the 2x2 matrix L_m?
    # No, the 2x2 Strassen algorithm takes two 2x2 matrices as input.
    # The input to the 2x2 Strassen for M1 is L1 and R1.
    # L1 is a 2x2 matrix. Its entries are linear forms on A.
    # R1 is a 2x2 matrix. Its entries are linear forms on B.
    
    # The 2x2 Strassen algorithm computes 7 products P1..P7.
    # P_j = (u_j . L1) * (v_j . R1).
    # Here u_j is a 4-dim vector, v_j is a 4-dim vector.
    # u_j . L1 means sum_{k=0}^3 u_j[k] * L1[k].
    # L1[k] is the k-th entry of the 2x2 matrix L1 (flattened).
    # L1[k] is a linear form on A.
    # So (u_j . L1) is a linear form on A.
    
    # So the scalar product triple is:
    # U_scalar = u_j . L1 (linear form on A)
    # V_scalar = v_j . R1 (linear form on B)
    # W_scalar = w_j (coeff for C entry)
    
    # We need to construct L1, R1 as 4-dim vectors of 16-dim vectors.
    # L1[k] is the k-th entry of the 2x2 matrix L1.
    
    # Let's define L1, R1 for each block product m.
    
    # M1: L1 = A00 + A11.
    # L1 is a 2x2 matrix.
    # L1[0,0] = A00[0,0] + A11[0,0]
    # L1[0,1] = A00[0,1] + A11[0,1]
    # L1[1,0] = A00[1,0] + A11[1,0]
    # L1[1,1] = A00[1,1] + A11[1,1]
    
    # We can represent L1 as a list of 4 16-dim vectors.
    
    def get_L_R_for_block_m(m):
        # Returns L (list of 4 16-dim vecs), R (list of 4 16-dim vecs)
        # m is 0..6
        # We define the 2x2 matrix structure for L and R.
        # Each entry is a sum of blocks.
        
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
        
        # M0 (M1): L = A00+A11, R = B00+B11
        if m == 0:
            L_blocks = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]] # Each entry is sum of block 0 and 3
            R_blocks = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
        elif m == 1: # M2
            L_blocks = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]] # A10+A11
            R_blocks = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]] # B00
        elif m == 2: # M3
            L_blocks = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]] # A00
            R_blocks = [[0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1]] # B01-B11
        elif m == 3: # M4
            L_blocks = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]] # A11
            R_blocks = [[-1, 0, 1, 0], [-1, 0, 1, 0], [-1, 0, 1, 0], [-1, 0, 1, 0]] # B10-B00
        elif m == 4: # M5
            L_blocks = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]] # A00+A01
            R_blocks = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]] # B11
        elif m == 5: # M6
            L_blocks = [[-1, 0, 1, 0], [-1, 0, 1, 0], [-1, 0, 1, 0], [-1, 0, 1, 0]] # A10-A00
            R_blocks = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]] # B00+B01
        else: # M7
            L_blocks = [[0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1], [0, 1, 0, -1]] # A01-A11
            R_blocks = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]] # B10+B11
            
        # Convert to list of 4 16-dim vectors
        L_vecs = []
        for k in range(4):
            vec = [0.0] * 16
            for b in range(4):
                coeff = L_blocks[k][b]
                if coeff != 0:
                    for e in range(4):
                        vec[b*4 + e] += coeff
            L_vecs.append(vec)
            
        R_vecs = []
        for k in range(4):
            vec = [0.0] * 16
            for b in range(4):
                coeff = R_blocks[k][b]
                if coeff != 0:
                    for e in range(4):
                        vec[b*4 + e] += coeff
            R_vecs.append(vec)
            
        return L_vecs, R_vecs

    triples = []
    for m in range(7):
        L_vecs, R_vecs = get_L_R_for_block_m(m)
        for j in range(7):
            # u_j, v_j from 2x2 Strassen
            u_j = u_2x2[j]
            v_j = v_2x2[j]
            
            # Compute U_scalar: sum_k u_j[k] * L_vecs[k]
            U_scalar = [0.0] * 16
            for k in range(4):
                if u_j[k] != 0:
                    for idx in range(16):
                        U_scalar[idx] += u_j[k] * L_vecs[k][idx]
                        
            # Compute V_scalar: sum_k v_j[k] * R_vecs[k]
            V_scalar = [0.0] * 16
            for k in range(4):
                if v_j[k] != 0:
                    for idx in range(16):
                        V_scalar[idx] += v_j[k] * R_vecs[k][idx]
                        
            # W_scalar: The contribution to the 4 entries of the 2x2 result matrix for this block product.
            # The 2x2 Strassen result is C_2x2 = R_2x2 @ M_2x2.
            # M_2x2 is the vector of 7 products.
            # We are at product j.
            # The contribution to C_2x2[0..3] is R_2x2[:, j] * P_j.
            # So W_scalar_2x2 = R_2x2[:, j].
            
            # Now we need to map this 2x2 result to the 4x4 result C.
            # The block product M_m contributes to specific 2x2 blocks of C.
            # We need the reconstruction for 4x4 Strassen.
            
            # 4x4 Strassen Reconstruction:
            # C00 = M1 - M4 + M6
            # C01 = M3 + M5
            # C10 = M2 + M7
            # C11 = M1 - M3 + M4
            
            # So for block product m, the coefficient for the block C_block is:
            # R_4x4[block_idx, m]
            
            # We need to know which block of C is being computed.
            # Wait, the 2x2 Strassen inside block product m computes a 2x2 matrix.
            # This 2x2 matrix IS the block M_m.
            # So the result of the 2x2 Strassen is the 4 entries of the block M_m.
            
            # Then, the 4x4 Strassen reconstruction combines the 7 blocks M1..M7 to form C00..C11.
            
            # So W_scalar for the 4x4 output is:
            # For each of the 4 entries of the 2x2 result (let's call them c0, c1, c2, c3 corresponding to block entries 0,1,2,3):
            # This entry contributes to specific entries of the 4x4 matrix C.
            
            # Let's define the 4x4 reconstruction matrix R_4x4_block.
            # R_4x4_block is a 4x7 matrix? No.
            # C has 4 blocks. Each block has 4 entries. Total 16 entries.
            # M has 7 blocks. Each block has 4 entries. Total 28 entries? No.
            # The 7 products M1..M7 are 2x2 matrices.
            # So we have 7 * 4 = 28 "block entries" from the products?
            # No, the Strassen algorithm computes 7 products of 2x2 matrices.
            # Each product is a 2x2 matrix.
            # So we have 7 * 4 = 28 scalar values from the products?
            # No, each "product" is a bilinear form that results in a 2x2 matrix?
            # No, in the recursive step, we treat the blocks as scalars.
            # So M1 is a scalar (in the ring of 2x2 matrices).
            # So M1 is a 2x2 matrix.
            # The entries of M1 are computed by 7 scalar products.
            
            # So for block product m, we have 7 scalar products P_{m,1}..P_{m,7}.
            # These 7 products form the 2x2 matrix M_m.
            # M_m[0,0] = R_2x2[0,0]*P_{m,1} + ...
            
            # Then C_block is a linear combination of M_1..M_7.
            # C_00 = M_1 - M_4 + M_6.
            # This means C_00[0,0] = M_1[0,0] - M_4[0,0] + M_6[0,0].
            
            # So for a specific scalar product P_{m,j}, it contributes to M_m[entry].
            # Which contributes to C_block[entry].
            
            # So W_scalar is a 16-dim vector.
            # For each of the 4 entries of the 2x2 result of the 2x2 Strassen (let's say entry e2 in {0,1,2,3}):
            # The coefficient is R_2x2[e2, j].
            # This entry e2 of M_m contributes to the 4 entries of the corresponding C block.
            # Which C block? It depends on m.
            
            # Let's define the 4x4 block reconstruction.
            # R_4x4 is a 4x7 matrix (4 C-blocks, 7 M-blocks).
            R_4x4 = np.array([
                [1, 0, 0, -1, 0, 1, 0], # C00
                [0, 0, 1, 0, 1, 0, 0],  # C01
                [0, 1, 0, 0, 0, 0, 1],  # C10
                [1, 0, -1, 1, 0, 0, 0]  # C11
            ])
            
            # For block product m, the contribution to C_block c is R_4x4[c, m].
            # The entry e2 of M_m contributes to the entry e2 of C_block c with coefficient R_4x4[c, m].
            
            W_scalar = [0.0] * 16
            for e2 in range(4):
                coeff_2x2 = R_2x2[e2, j]
                if coeff_2x2 == 0: continue
                for c_block in range(4):
                    coeff_4x4 = R_4x4[c_block, m]
                    if coeff_4x4 == 0: continue
                    total_coeff = coeff_2x2 * coeff_4x4
                    # Map c_block and e2 to 16-dim index
                    idx = c_block * 4 + e2
                    W_scalar[idx] += total_coeff
                    
            triples.append((U_scalar, V_scalar, W_scalar))
            
    return triples

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    triples_4x4 = get_strassen_4x4_triples()
    
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill U, V, W for the first 49 (B_L)
    for r in range(49):
        u, v, w = triples_4x4[r]
        # u is 16-dim (A)
        # v is 16-dim (B_L, which is 4x4)
        # w is 16-dim (C_L, which is 4x4)
        
        for i in range(16):
            U[i][r] = u[i]
            
        # Map v (16-dim, j in 0..3, k' in 0..3) to V (32-dim, j in 0..3, k in 0..7)
        for j in range(4):
            for k_prime in range(4):
                V[j * 8 + k_prime][r] = v[j * 4 + k_prime]
                
        # Map w (16-dim, k' in 0..3, i in 0..3) to W (32-dim, k in 0..7, i in 0..3)
        # W index is k * n + i = k * 4 + i
        for k_prime in range(4):
            for i in range(4):
                W[k_prime * 4 + i][r] = w[k_prime * 4 + i]
                
    # Fill U, V, W for the next 49 (B_R)
    for r in range(49):
        u, v, w = triples_4x4[r]
        r_full = 49 + r
        
        for i in range(16):
            U[i][r_full] = u[i]
            
        # Map v to V for B_R (k in 4..7)
        for j in range(4):
            for k_prime in range(4):
                V[j * 8 + (k_prime + 4)][r_full] = v[j * 4 + k_prime]
                
        # Map w to W for C_R (k in 4..7)
        for k_prime in range(4):
            for i in range(4):
                W[(k_prime + 4) * 4 + i][r_full] = w[k_prime * 4 + i]
                
    return {"U": U, "V": V, "W": W}
