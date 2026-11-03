"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + Rank 48 for 4x4x3 (Schoolbook) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # Columns 0-3 of B, Columns 0-3 of C
    # Indices in V: 0-15 (4x4)
    # Indices in W: 0-15 (4x4)
    
    # Define the 7 Strassen steps for 2x2x2
    # U_local: [a, b, c, d]
    # V_local: [e, f, g, h]
    # W_local: [p, q, r, s]
    
    u_base = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    v_base = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    w_base = [
        [1, 0, 0, 1], # M1: p+s
        [0, 0, 1, -1],# M2: r-s
        [0, 1, 0, 1], # M3: q+s
        [1, 0, 1, 0], # M4: p+r
        [-1, 1, 0, 0],# M5: -p+q
        [0, 0, 0, 1], # M6: s
        [1, 0, 0, 0]  # M7: p
    ]
    
    # Mapping of 2x2 blocks to 4x4 indices
    # A blocks:
    # A11: 0,1,4,5 -> map to 0,1,2,3
    # A12: 2,3,6,7 -> map to 0,1,2,3
    # A21: 8,9,12,13 -> map to 0,1,2,3
    # A22: 10,11,14,15 -> map to 0,1,2,3
    
    def map_u(val_a11, val_a12, val_a21, val_a22):
        # val_x are 4-vectors for the 2x2 block
        vec = [0] * 16
        for i in range(4):
            vec[0 + i] = val_a11[i] if i < 4 else 0 # Incorrect mapping logic below
        # Correct Mapping:
        # 0,1,4,5 are A11. Local idx 0,1,2,3 correspond to 0,1,4,5.
        vec[0] = val_a11[0]
        vec[1] = val_a11[1]
        vec[4] = val_a11[2]
        vec[5] = val_a11[3]
        
        vec[2] = val_a12[0]
        vec[3] = val_a12[1]
        vec[6] = val_a12[2]
        vec[7] = val_a12[3]
        
        vec[8] = val_a21[0]
        vec[9] = val_a21[1]
        vec[12] = val_a21[2]
        vec[13] = val_a21[3]
        
        vec[10] = val_a22[0]
        vec[11] = val_a22[1]
        vec[14] = val_a22[2]
        vec[15] = val_a22[3]
        return vec

    def map_v(val_b11, val_b12, val_b21, val_b22):
        vec = [0] * 16
        # B is 4x4. Columns 0-3.
        # B11: cols 0-1 of B, rows 0-1. Indices 0,1, 4,5?
        # V index is j*p + k. j is row of A/col of C? No.
        # V is (m x p). m=4 (rows of B? No, columns of A?).
        # Convention: U[n*m, rank], V[m*p, rank], W[p*n, rank].
        # U[i*m+j], V[j*p+k], W[k*n+i].
        # Sum U[i,m,j] V[j,p,k] W[k,n,i] = C[i,p,k]?
        # Standard: A is n x m. B is m x p. C is n x p.
        # Here Task says: 4x4 by 4x7.
        # A: 4x4. B: 4x7. C: 4x7.
        # n=4, m=4, p=7.
        # U: 4x4. V: 4x7. W: 7x4.
        # U[i*m+j] = A[i,j].
        # V[j*p+k] = B[j,k].
        # W[k*n+i] = C[i,k].
        
        # So for V (4x4 part, k=0..3):
        # j=0: k=0,1,2,3 -> V[0], V[1], V[2], V[3]
        # j=1: k=0,1,2,3 -> V[4], V[5], V[6], V[7]
        # j=2: k=0,1,2,3 -> V[8], V[9], V[10], V[11]
        # j=3: k=0,1,2,3 -> V[12], V[13], V[14], V[15]
        
        # B blocks (4x4):
        # B11: rows 0-1, cols 0-1.
        # B12: rows 0-1, cols 2-3.
        # B21: rows 2-3, cols 0-1.
        # B22: rows 2-3, cols 2-3.
        
        # Map to V indices:
        # B11: V[0], V[1], V[4], V[5] (local 0,1,2,3 -> 0,1,4,5)
        # B12: V[2], V[3], V[6], V[7] (local 0,1,2,3 -> 2,3,6,7)
        # B21: V[8], V[9], V[12], V[13] (local 0,1,2,3 -> 8,9,12,13)
        # B22: V[10], V[11], V[14], V[15] (local 0,1,2,3 -> 10,11,14,15)
        
        vec[0] = val_b11[0]
        vec[1] = val_b11[1]
        vec[4] = val_b11[2]
        vec[5] = val_b11[3]
        
        vec[2] = val_b12[0]
        vec[3] = val_b12[1]
        vec[6] = val_b12[2]
        vec[7] = val_b12[3]
        
        vec[8] = val_b21[0]
        vec[9] = val_b21[1]
        vec[12] = val_b21[2]
        vec[13] = val_b21[3]
        
        vec[10] = val_b22[0]
        vec[11] = val_b22[1]
        vec[14] = val_b22[2]
        vec[15] = val_b22[3]
        return vec

    def map_w(val_c11, val_c12, val_c21, val_c22):
        vec = [0] * 16
        # W index k*n + i.
        # C is 4x4 (cols 0-3).
        # C11: rows 0-1, cols 0-1.
        # C12: rows 0-1, cols 2-3.
        # C21: rows 2-3, cols 0-1.
        # C22: rows 2-3, cols 2-3.
        
        # k=0 (col 0):
        # C11 uses i=0,1. W[0], W[1].
        # C21 uses i=2,3. W[8], W[9].
        # k=1 (col 1):
        # C11 uses i=0,1. W[4], W[5].
        # C21 uses i=2,3. W[12], W[13].
        # k=2 (col 2):
        # C12 uses i=0,1. W[2], W[3].
        # C22 uses i=2,3. W[10], W[11].
        # k=3 (col 3):
        # C12 uses i=0,1. W[6], W[7].
        # C22 uses i=2,3. W[14], W[15].
        
        # Local c11: [i0c0, i1c0, i0c1, i1c1]? 
        # My local W definition was [p,q,r,s] corresponding to C[0,0], C[0,1], C[1,0], C[1,1].
        # So w_base[k] is for C[k_row, k_col]?
        # In Strassen derivation:
        # C11 is top-left. C[0,0], C[0,1], C[1,0], C[1,1].
        # Local indices 0,1,2,3 map to (0,0), (0,1), (1,0), (1,1).
        
        # W[0*n + 0] = W[0] gets C[0,0] -> local 0
        # W[1*n + 0] = W[4] gets C[1,0] -> local 2
        # W[2*n + 0] = W[8] gets C[2,0] -> Part of C21
        # W[3*n + 0] = W[12] gets C[3,0] -> Part of C21
        
        # C11 (Top-Left 2x2):
        # C[0,0] -> W[0]
        # C[1,0] -> W[4]
        # C[0,1] -> W[1]
        # C[1,1] -> W[5]
        
        # C12 (Top-Right 2x2):
        # C[0,2] -> W[2]
        # C[1,2] -> W[6]
        # C[0,3] -> W[3]
        # C[1,3] -> W[7]
        
        # C21 (Bottom-Left 2x2):
        # C[2,0] -> W[8]
        # C[3,0] -> W[12]
        # C[2,1] -> W[9]
        # C[3,1] -> W[13]
        
        # C22 (Bottom-Right 2x2):
        # C[2,2] -> W[10]
        # C[3,2] -> W[14]
        # C[2,3] -> W[11]
        # C[3,3] -> W[15]
        
        vec[0] = val_c11[0]
        vec[4] = val_c11[2]
        vec[1] = val_c11[1]
        vec[5] = val_c11[3]
        
        vec[2] = val_c12[0]
        vec[6] = val_c12[2]
        vec[3] = val_c12[1]
        vec[7] = val_c12[3]
        
        vec[8] = val_c21[0]
        vec[12] = val_c21[2]
        vec[9] = val_c21[1]
        vec[13] = val_c21[3]
        
        vec[10] = val_c22[0]
        vec[14] = val_c22[2]
        vec[11] = val_c22[1]
        vec[15] = val_c22[3]
        
        return vec

    zero_vec = [0,0,0,0]
    
    # Define which blocks are used for each Strassen term
    # Term 1: (A11+A22)(B11+B22) -> C11, C22
    u1 = [u_base[0], zero_vec, zero_vec, u_base[0]]
    v1 = [v_base[0], zero_vec, zero_vec, v_base[0]]
    w1 = [w_base[0], zero_vec, zero_vec, w_base[0]] # C11 gets w, C22 gets w
    
    # Term 2: (A21+A22)B11 -> C21, C22
    u2 = [zero_vec, zero_vec, u_base[1], u_base[1]] # Wait, A21 is c, A22 is d. M2=(c+d)e.
    # u_base[1] is [0,0,1,1] (c,d).
    # So A21 gets [0,0,1,1]? No.
    # A21 is a block. The values in the block are determined by the local vector.
    # For M2, the local vector is applied to A21 and A22?
    # Strassen M2 = (A21 + A22) * B11.
    # This is (Sum of blocks) * Block.
    # This is not a simple tensor product of two separate blocks.
    # It is a linear combination.
    # The term contributes to U as: U_vec = U_A21 + U_A22.
    # Where U_A21 is the expansion of the vector for A21, and U_A22 for A22.
    
    # Let's restart the mapping logic more carefully.
    # For each Strassen term t (0..6):
    # It has a U_local vector, V_local vector, W_local vector.
    # It acts on specific blocks.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U part: A11 and A22 are multiplied by 1 (from u_base[0]).
    # V part: B11 and B22 are multiplied by 1 (from v_base[0]).
    # W part: C11 and C22 receive contributions.
    
    # Term 2: M2 = (A21+A22)B11
    # U part: A21 and A22.
    # V part: B11.
    # W part: C21 and C22.
    
    # Term 3: M3 = A11(B12-B22)
    # U part: A11.
    # V part: B12 and B22.
    # W part: C12 and C22.
    
    # Term 4: M4 = A22(B21-B11)
    # U part: A22.
    # V part: B21 and B11.
    # W part: C11 and C21.
    
    # Term 5: M5 = (A11+A12)B22
    # U part: A11 and A12.
    # V part: B22.
    # W part: C11 and C12.
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # U part: A21 and A11.
    # V part: B11 and B12.
    # W part: C22.
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # U part: A12 and A22.
    # V part: B21 and B22.
    # W part: C11.

    # We will construct the 49 rank vectors by iterating the 7 terms.
    # Each term t uses 7 basis vectors from u_base, v_base, w_base.
    # So 7 terms * 7 basis = 49 vectors.
    
    # Structure:
    # for t in range(7):
    #   for b in range(7):
    #     U_vec = expand_u(u_base[b], block_mask_u[t])
    #     V_vec = expand_v(v_base[b], block_mask_v[t])
    #     W_vec = expand_w(w_base[b], block_mask_w[t])
    
    # We need to define the block masks and coefficients.
    
    # U Blocks: 11, 12, 21, 22
    # V Blocks: 11, 12, 21, 22
    # W Blocks: 11, 12, 21, 22
    
    # t=0 (M1): U{11:1, 22:1}, V{11:1, 22:1}, W{11:1, 22:1}
    # t=1 (M2): U{21:1, 22:1}, V{11:1}, W{21:1, 22:-1}
    # t=2 (M3): U{11:1}, V{12:1, 22:-1}, W{12:1, 22:1}
    # t=3 (M4): U{22:1}, V{21:1, 11:-1}, W{11:1, 21:1}
    # t=4 (M5): U{11:1, 12:1}, V{22:1}, W{11:-1, 12:1}
    # t=5 (M6): U{21:1, 11:-1}, V{11:1, 12:1}, W{22:1}
    # t=6 (M7): U{12:1, 22:-1}, V{21:1, 22:1}, W{11:1}
    
    u_blocks = [
        {11:1, 22:1},
        {21:1, 22:1},
        {11:1},
        {22:1},
        {11:1, 12:1},
        {21:1, 11:-1},
        {12:1, 22:-1}
    ]
    
    v_blocks = [
        {11:1, 22:1},
        {11:1},
        {12:1, 22:-1},
        {21:1, 11:-1},
        {22:1},
        {11:1, 12:1},
        {21:1, 22:1}
    ]
    
    w_blocks = [
        {11:1, 22:1},
        {21:1, 22:-1},
        {12:1, 22:1},
        {11:1, 21:1},
        {11:-1, 12:1},
        {22:1},
        {11:1}
    ]

    def expand_u(vec, mask):
        a11 = [x * mask.get(11, 0) for x in vec]
        a12 = [x * mask.get(12, 0) for x in vec]
        a21 = [x * mask.get(21, 0) for x in vec]
        a22 = [x * mask.get(22, 0) for x in vec]
        return map_u(a11, a12, a21, a22)

    def expand_v(vec, mask):
        b11 = [x * mask.get(11, 0) for x in vec]
        b12 = [x * mask.get(12, 0) for x in vec]
        b21 = [x * mask.get(21, 0) for x in vec]
        b22 = [x * mask.get(22, 0) for x in vec]
        return map_v(b11, b12, b21, b22)

    def expand_w(vec, mask):
        c11 = [x * mask.get(11, 0) for x in vec]
        c12 = [x * mask.get(12, 0) for x in vec]
        c21 = [x * mask.get(21, 0) for x in vec]
        c22 = [x * mask.get(22, 0) for x in vec]
        return map_w(c11, c12, c21, c22)

    col = 0
    for t in range(7):
        for b in range(7):
            U[:, col] = expand_u(u_base[b], u_blocks[t])
            V[:, col] = expand_v(v_base[b], v_blocks[t])
            W[:, col] = expand_w(w_base[b], w_blocks[t])
            col += 1
            
    # --- Part 2: 4x4x3 Schoolbook (Rank 48) ---
    # Columns 4-6 of B, Columns 4-6 of C
    
    # V indices: j*7 + k_global. k_global in 4,5,6.
    # W indices: k_global*4 + i.
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4 + j][col] = 1
                V[j*7 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
