"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 (Strassen for 4x4)
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base Strassen for 2x2
    # Returns list of 7 tuples (u_vec, v_vec, w_vec) for 2x2 indices
    # u_vec, v_vec, w_vec are length 4
    def base_strassen_2x2():
        terms = []
        # m1 = (a+d)(e+h)
        # a=0, b=1, c=2, d=3
        # e=0, f=1, g=2, h=3
        # u: a+d -> [1, 0, 0, 1]
        # v: e+h -> [1, 0, 0, 1]
        # w: C00 (+1), C11 (+1) -> indices 0, 3 -> [1, 0, 0, 1]
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # m2 = (c+d)e
        # u: c+d -> [0, 0, 1, 1]
        # v: e   -> [1, 0, 0, 0]
        # w: C10 (+1) -> index 2 -> [0, 0, 1, 0]
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        
        # m3 = a(f-h)
        # u: a   -> [1, 0, 0, 0]
        # v: f-h -> [0, 1, 0, -1]
        # w: C01 (+1) -> index 1 -> [0, 1, 0, 0]
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        
        # m4 = d(g-e)
        # u: d   -> [0, 0, 0, 1]
        # v: g-e -> [0, 0, 1, -1] # Wait, g is index 2, e is index 0. g-e -> [ -1, 0, 1, 0]
        # v: g-e -> [-1, 0, 1, 0]
        # w: C00 (+1), C10 (+1) -> indices 0, 2 -> [1, 0, 1, 0]
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        
        # m5 = (a+b)h
        # u: a+b -> [1, 1, 0, 0]
        # v: h   -> [0, 0, 0, 1]
        # w: C00 (-1), C01 (+1) -> indices 0, 1 -> [-1, 1, 0, 0]
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        
        # m6 = (c-a)(e+f)
        # u: c-a -> [-1, 0, 1, 0]
        # v: e+f -> [1, 1, 0, 0]
        # w: C11 (+1) -> index 3 -> [0, 0, 0, 1]
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # m7 = (b-d)(g+h)
        # u: b-d -> [0, 1, 0, -1]
        # v: g+h -> [0, 0, 1, 1]
        # w: C00 (+1), C11 (-1) -> indices 0, 3 -> [1, 0, 0, -1]
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]))
        
        return terms

    base_terms = base_strassen_2x2()
    
    # Top level Strassen for 4x4 (2x2 blocks of 2x2)
    # Blocks:
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each M_k, we define the linear combinations of the 4 blocks.
    # Each block is 2x2.
    # We need to map the 2x2 local indices (0,1,2,3) to the 4x4 global indices.
    
    # Local index mapping for a block:
    # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1) relative to block top-left.
    
    def get_block_indices(block_id):
        # block_id: 0=A11, 1=A12, 2=A21, 3=A22
        # Returns list of 4 global indices for U (i*m+j)
        # i, j are 0..3
        if block_id == 0: # A11
            return [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
        elif block_id == 1: # A12
            return [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7
        elif block_id == 2: # A21
            return [2*4+0, 2*4+1, 3*4+0, 3*4+1] # 8, 9, 12, 13
        elif block_id == 3: # A22
            return [2*4+2, 2*4+3, 3*4+2, 3*4+3] # 10, 11, 14, 15

    # V indices (j*p+k)
    def get_block_indices_V(block_id):
        if block_id == 0: # B11
            return [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
        elif block_id == 1: # B12
            return [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7
        elif block_id == 2: # B21
            return [2*4+0, 2*4+1, 3*4+0, 3*4+1] # 8, 9, 12, 13
        elif block_id == 3: # B22
            return [2*4+2, 2*4+3, 3*4+2, 3*4+3] # 10, 11, 14, 15

    # W indices (k*n+i)
    def get_block_indices_W(block_id):
        # C11: i in 0,1; k in 0,1
        # C12: i in 0,1; k in 2,3
        # C21: i in 2,3; k in 0,1
        # C22: i in 2,3; k in 2,3
        # Local 0->(k=0,i=0), 1->(k=0,i=1)? No.
        # W row index is k*n + i.
        # For C11 (i<2, k<2):
        # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3?
        # Wait, local index 0 is (0,0) in block.
        # If block is C11, top-left is (i=0, k=0).
        # Local 0: i=0, k=0 -> 0*4+0=0
        # Local 1: i=1, k=0 -> 0*4+1=1
        # Local 2: i=0, k=1 -> 1*4+0=4
        # Local 3: i=1, k=1 -> 1*4+1=5
        # This mapping depends on the block position.
        
        if block_id == 0: # C11
            return [0*4+0, 0*4+1, 1*4+0, 1*4+1] # 0, 1, 4, 5
        elif block_id == 1: # C12
            return [2*4+0, 2*4+1, 3*4+0, 3*4+1] # 8, 9, 12, 13
        elif block_id == 2: # C21
            return [0*4+2, 0*4+3, 1*4+2, 1*4+3] # 2, 3, 6, 7
        elif block_id == 3: # C22
            return [2*4+2, 2*4+3, 3*4+2, 3*4+3] # 10, 11, 14, 15

    # Define the 7 top-level terms
    # Each term is a list of (coeff, block_id) for A, B, and C
    # A_blocks, B_blocks, C_blocks
    top_terms = [
        # M1
        ([(1, 0), (1, 3)], [(1, 0), (1, 3)], [(1, 0), (1, 3)]),
        # M2
        ([(1, 2), (1, 3)], [(1, 0)], [(1, 2)]),
        # M3
        ([(1, 0)], [(1, 1), (-1, 3)], [(1, 1)]),
        # M4
        ([(1, 3)], [(1, 2), (-1, 0)], [(1, 0), (1, 2)]),
        # M5
        ([(1, 0), (1, 1)], [(1, 3)], [(-1, 0), (1, 1)]),
        # M6
        ([(1, 2), (-1, 0)], [(1, 0), (1, 1)], [(1, 3)]),
        # M7
        ([(1, 1), (-1, 3)], [(1, 2), (1, 3)], [(1, 0), (-1, 3)])
    ]
    
    # Iterate through top-level terms
    for g, (a_comb, b_comb, c_comb) in enumerate(top_terms):
        # For each base term in this group
        for l, (u_base, v_base, w_base) in enumerate(base_terms):
            r = g * 7 + l
            
            # Construct U vector for this rank
            # u_base has 4 entries corresponding to local block indices 0,1,2,3
            # We need to sum the contributions from all blocks in a_comb
            u_vec = [0] * 16
            for coeff, block_id in a_comb:
                indices = get_block_indices(block_id)
                for local_idx in range(4):
                    global_idx = indices[local_idx]
                    u_vec[global_idx] += coeff * u_base[local_idx]
            
            # Construct V vector
            v_vec = [0] * 16
            for coeff, block_id in b_comb:
                indices = get_block_indices_V(block_id)
                for local_idx in range(4):
                    global_idx = indices[local_idx]
                    v_vec[global_idx] += coeff * v_base[local_idx]
            
            # Construct W vector
            w_vec = [0] * 16
            for coeff, block_id in c_comb:
                indices = get_block_indices_W(block_id)
                for local_idx in range(4):
                    global_idx = indices[local_idx]
                    w_vec[global_idx] += coeff * w_base[local_idx]
            
            # Fill U, V, W
            for i in range(16):
                U[i][r] = u_vec[i]
                V[i][r] = v_vec[i]
                W[i][r] = w_vec[i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
