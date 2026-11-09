import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114 decomposition
    rank = 114
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        # u_vec is length n*m
        # v_vec is length m*p
        # w_vec is length p*n
        U[:, col_idx] = u_vec
        V[:, col_idx] = v_vec
        W[:, col_idx] = w_vec

    # Strassen 2x2x2 factors
    # u, v, w are length 4
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ]
    
    # Coefficients for A blocks in each Strassen term
    # A blocks: 00, 01, 10, 11 (indices 0,1,2,3 in 2x2 grid)
    # u_s selects blocks.
    # We need to map the 2x2x2 decomposition to 4x4x4.
    
    # This is effectively a Kronecker product of the 2x2x2 decomposition with itself?
    # No, it's the Strassen recursion.
    
    # Let's generate the 49 terms for 4x4x4.
    # We iterate over the 7 Strassen terms of the outer level.
    # For each, we iterate over the 7 Strassen terms of the inner level.
    
    col = 0
    for r in range(7):
        # Outer term r
        # Determine which A blocks are involved and with what coefficients
        # Based on s_u[r]
        a_coeff = [0.0, 0.0, 0.0, 0.0]
        a_coeff[0] = s_u[r][0] # A00
        a_coeff[1] = s_u[r][1] # A01
        a_coeff[2] = s_u[r][2] # A10
        a_coeff[3] = s_u[r][3] # A11
        
        b_coeff = [0.0, 0.0, 0.0, 0.0]
        b_coeff[0] = s_v[r][0] # B00
        b_coeff[1] = s_v[r][1] # B01
        b_coeff[2] = s_v[r][2] # B10
        b_coeff[3] = s_v[r][3] # B11
        
        # c_coeff for output blocks
        c_coeff = [0.0, 0.0, 0.0, 0.0]
        c_coeff[0] = s_w[r][0] # C00
        c_coeff[1] = s_w[r][1] # C01
        c_coeff[2] = s_w[r][2] # C10
        c_coeff[3] = s_w[r][3] # C11
        
        for s in range(7):
            # Inner term s
            # This term computes a 2x2x2 multiplication.
            # The inputs are the linear combinations defined by a_coeff and b_coeff?
            # No.
            # The outer Strassen step says:
            # M_r = (Sum a_i A_i) * (Sum b_j B_j)
            # This is a 2x2 matrix multiplication.
            # We decompose this 2x2 multiplication using Strassen (inner level).
            
            # The inner level operates on the "virtual" 2x2 matrices.
            # The "virtual" A matrix is A_r = Sum a_i A_i.
            # The "virtual" B matrix is B_r = Sum b_j B_j.
            
            # The inner Strassen decomposition of A_r * B_r uses 7 scalar mults.
            # For each scalar mult s:
            # u_s selects elements of A_r.
            # v_s selects elements of B_r.
            # w_s selects elements of the result M_r.
            
            # So we need to expand u_s, v_s, w_s back to the original 4x4 indices.
            
            # u_s is a vector of 4 coefficients for the 2x2 blocks of A_r.
            # A_r block 0 is Sum a_i A_i (where A_i is 2x2 block).
            # So u_s[0] * A_r[0] = u_s[0] * (a_0 A_00 + a_1 A_01 + a_2 A_10 + a_3 A_11)
            
            # We need to construct the full U vector (length 16).
            u_full = np.zeros(16)
            v_full = np.zeros(36) # 4x9
            w_full = np.zeros(36) # 9x4
            
            # Map A blocks to indices in U (4x4)
            # Block 0 (00): rows 0,1; cols 0,1. Indices 0,1,4,5
            # Block 1 (01): rows 0,1; cols 2,3. Indices 2,3,6,7
            # Block 2 (10): rows 2,3; cols 0,1. Indices 8,9,12,13
            # Block 3 (11): rows 2,3; cols 2,3. Indices 10,11,14,15
            
            # For each element in u_s (index 0..3):
            for blk in range(4):
                coeff = u_s[s][blk] * a_coeff[blk]
                if coeff != 0:
                    # Add coeff to the 4 indices of this block in U
                    # Block 0: 0,1,4,5
                    # Block 1: 2,3,6,7
                    # Block 2: 8,9,12,13
                    # Block 3: 10,11,14,15
                    # We need the mapping from 2x2 block index to 4 indices.
                    # Block (br, bc) where br, bc in 0..1
                    # Indices: br*4 + bc, br*4 + bc + 1, br*4 + 4 + bc, br*4 + 4 + bc + 1
                    # Wait, U index is i*m + j.
                    # Block 0 (br=0, bc=0): i=0,1; j=0,1.
                    # i=0, j=0 -> 0
                    # i=0, j=1 -> 1
                    # i=1, j=0 -> 4
                    # i=1, j=1 -> 5
                    # Block 1 (br=0, bc=1): i=0,1; j=2,3.
                    # i=0, j=2 -> 2
                    # i=0, j=3 -> 3
                    # i=1, j=2 -> 6
                    # i=1, j=3 -> 7
                    # Block 2 (br=1, bc=0): i=2,3; j=0,1.
                    # i=2, j=0 -> 8
                    # i=2, j=1 -> 9
                    # i=3, j=0 -> 12
                    # i=3, j=1 -> 13
                    # Block 3 (br=1, bc=1): i=2,3; j=2,3.
                    # i=2, j=2 -> 10
                    # i=2, j=3 -> 11
                    # i=3, j=2 -> 14
                    # i=3, j=3 -> 15
                    
                    br = blk // 2
                    bc = blk % 2
                    idxs = [
                        br*4 + bc,
                        br*4 + bc + 1,
                        (br+1)*4 + bc,
                        (br+1)*4 + bc + 1
                    ]
                    for idx in idxs:
                        u_full[idx] += coeff

            # Similarly for V (4x9)
            # B blocks are 2x2.
            # Block 0 (00): rows 0,1; cols 0,1.
            # V index is j*p + k.
            # Block 0: j=0,1; k=0,1.
            # j=0, k=0 -> 0
            # j=0, k=1 -> 1
            # j=1, k=0 -> 9
            # j=1, k=1 -> 10
            # Block 1 (01): j=0,1; k=2,3.
            # j=0, k=2 -> 2
            # j=0, k=3 -> 3
            # j=1, k=2 -> 11
            # j=1, k=3 -> 12
            # Block 2 (10): j=2,3; k=0,1.
            # j=2, k=0 -> 18
            # j=2, k=1 -> 19
            # j=3, k=0 -> 27
            # j=3, k=1 -> 28
            # Block 3 (11): j=2,3; k=2,3.
            # j=2, k=2 -> 20
            # j=2, k=3 -> 21
            # j=3, k=2 -> 29
            # j=3, k=3 -> 30
            
            for blk in range(4):
                coeff = v_s[s][blk] * b_coeff[blk]
                if coeff != 0:
                    br = blk // 2
                    bc = blk % 2
                    # j start: br*2
                    # k start: bc*2
                    j0 = br * 2
                    k0 = bc * 2
                    idxs = [
                        j0 * 9 + k0,
                        j0 * 9 + k0 + 1,
                        (j0+1) * 9 + k0,
                        (j0+1) * 9 + k0 + 1
                    ]
                    for idx in idxs:
                        v_full[idx] += coeff

            # Similarly for W (9x4)
            # W index is k*n + i.
            # C blocks are 2x2.
            # Block 0 (00): k=0,1; i=0,1.
            # k=0, i=0 -> 0
            # k=0, i=1 -> 1
            # k=1, i=0 -> 4
            # k=1, i=1 -> 5
            # Block 1 (01): k=0,1; i=2,3.
            # k=0, i=2 -> 2
            # k=0, i=3 -> 3
            # k=1, i=2 -> 6
            # k=1, i=3 -> 7
            # Block 2 (10): k=2,3; i=0,1.
            # k=2, i=0 -> 8
            # k=2, i=1 -> 9
            # k=3, i=0 -> 12
            # k=3, i=1 -> 13
            # Block 3 (11): k=2,3; i=2,3.
            # k=2, i=2 -> 10
            # k=2, i=3 -> 11
            # k=3, i=2 -> 14
            # k=3, i=3 -> 15
            
            for blk in range(4):
                coeff = w_s[s][blk] * c_coeff[blk]
                if coeff != 0:
                    br = blk // 2 # k block row
                    bc = blk % 2 # i block col
                    k0 = br * 2
                    i0 = bc * 2
                    idxs = [
                        k0 * 4 + i0,
                        k0 * 4 + i0 + 1,
                        (k0+1) * 4 + i0,
                        (k0+1) * 4 + i0 + 1
                    ]
                    for idx in idxs:
                        w_full[idx] += coeff

            add_term(u_full, v_full, w_full, col)
            col += 1
            
    # Now col is 49.
    # We need to repeat this for the second 4x4x4 block (cols 4-7 of B).
    # The U and W are the same.
    # The V indices shift by 4 in k.
    
    for r in range(7):
        a_coeff = [0.0, 0.0, 0.0, 0.0]
        a_coeff[0] = s_u[r][0]
        a_coeff[1] = s_u[r][1]
        a_coeff[2] = s_u[r][2]
        a_coeff[3] = s_u[r][3]
        
        b_coeff = [0.0, 0.0, 0.0, 0.0]
        b_coeff[0] = s_v[r][0]
        b_coeff[1] = s_v[r][1]
        b_coeff[2] = s_v[r][2]
        b_coeff[3] = s_v[r][3]
        
        c_coeff = [0.0, 0.0, 0.0, 0.0]
        c_coeff[0] = s_w[r][0]
        c_coeff[1] = s_w[r][1]
        c_coeff[2] = s_w[r][2]
        c_coeff[3] = s_w[r][3]
        
        for s in range(7):
            u_full = np.zeros(16)
            v_full = np.zeros(36)
            w_full = np.zeros(36)
            
            for blk in range(4):
                coeff = u_s[s][blk] * a_coeff[blk]
                if coeff != 0:
                    br = blk // 2
                    bc = blk % 2
                    idxs = [
                        br*4 + bc,
                        br*4 + bc + 1,
                        (br+1)*4 + bc,
                        (br+1)*4 + bc + 1
                    ]
                    for idx in idxs:
                        u_full[idx] += coeff

            for blk in range(4):
                coeff = v_s[s][blk] * b_coeff[blk]
                if coeff != 0:
                    br = blk // 2
                    bc = blk % 2
                    j0 = br * 2
                    k0 = bc * 2 + 4 # Shift by 4
                    idxs = [
                        j0 * 9 + k0,
                        j0 * 9 + k0 + 1,
                        (j0+1) * 9 + k0,
                        (j0+1) * 9 + k0 + 1
                    ]
                    for idx in idxs:
                        v_full[idx] += coeff

            for blk in range(4):
                coeff = w_s[s][blk] * c_coeff[blk]
                if coeff != 0:
                    br = blk // 2
                    bc = blk % 2
                    k0 = br * 2 + 4 # Shift by 4
                    i0 = bc * 2
                    idxs = [
                        k0 * 4 + i0,
                        k0 * 4 + i0 + 1,
                        (k0+1) * 4 + i0,
                        (k0+1) * 4 + i0 + 1
                    ]
                    for idx in idxs:
                        w_full[idx] += coeff

            add_term(u_full, v_full, w_full, col)
            col += 1
            
    # Now col is 98.
    # Add 4x4x1 block (col 8 of B).
    # 16 scalar multiplications.
    for i in range(4):
        for j in range(4):
            u_full = np.zeros(16)
            v_full = np.zeros(36)
            w_full = np.zeros(36)
            
            u_full[i*4 + j] = 1.0
            v_full[j*9 + 8] = 1.0
            w_full[8*4 + i] = 1.0
            
            add_term(u_full, v_full, w_full, col)
            col += 1
            
    return {"U": U, "V": V, "W": W}
