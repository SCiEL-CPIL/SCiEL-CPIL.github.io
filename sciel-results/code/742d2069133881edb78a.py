def construct():
    n, m, p = 3, 3, 4
    # Rank calculation:
    # 2x2x4 part: 2 * 7 = 14
    # 2x1x4 part: 1
    # 1x2x4 part: 1
    # 1x1x4 part: 1
    # Total rank = 17
    
    r = 17
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]

    # Helper to add a term to the decomposition
    # u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        U[u_idx][r_current] += coeff
        V[v_idx][r_current] += coeff
        W[w_idx][r_current] += coeff

    # We need to manage the rank index.
    # It's easier to pre-allocate and fill specific columns.
    
    # Let's re-initialize with a list of terms.
    terms = []
    
    # Sub-problem 1: A_11 (2x2) * B_11 (2x4) -> C_11 (2x4)
    # A_11: U indices 0, 1, 3, 4
    # B_11: V indices 0, 1, 2, 3 (col 0 of B) and 4, 5, 6, 7 (col 1 of B? No, B is 2x4)
    # B_11 is rows 0,1 of B. So V indices 0..3 (j=0) and 4..7 (j=1).
    # Wait, B is 3x4.
    # V index = j*p + k.
    # j=0: 0,1,2,3.
    # j=1: 4,5,6,7.
    # j=2: 8,9,10,11.
    
    # B_11 corresponds to j=0,1.
    # For the first 2x2x2 (B_11[:, 0:2]), B indices are:
    # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[4], h=B[1,1]=V[5].
    # For the second 2x2x2 (B_11[:, 2:4]), B indices are:
    # e=B[0,2]=V[2], f=B[0,3]=V[3], g=B[1,2]=V[6], h=B[1,3]=V[7].
    
    # A_11 indices:
    # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[3], d=A[1,1]=U[4].
    
    # C_11 indices (W):
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4].
    # C[0,2]=W[6], C[0,3]=W[7], C[1,2]=W[9], C[1,3]=W[10].
    
    # Strassen terms for first block (cols 0,1 of B)
    # M1 = (a+d)(e+h)
    # Terms: (a*e, a*h, d*e, d*h)
    # This is a rank 4 expansion? No, Strassen is rank 7.
    # M1 = (a+d)(e+h) = a(e+h) + d(e+h).
    # This is not a single rank 1 term.
    # The Strassen decomposition is:
    # M1 = (a+d)(e+h) -> U=(1,0,1,0), V=(1,1,0,0), W=(1,0,0,1) ? No.
    # The tensor decomposition for Strassen is:
    # T = M1*(e1+e4)(e1+e2) + ...
    # Actually, it's easier to just list the 7 rank-1 tensors.
    
    # Let's use the standard Strassen rank-1 tensors for 2x2x2.
    # Indices for 2x2 A: 0,1,2,3 (row-major: 00,01,10,11)
    # Indices for 2x2 B: 0,1,2,3 (row-major: 00,01,10,11)
    # Indices for 2x2 C: 0,1,2,3 (row-major: 00,01,10,11)
    
    # Strassen tensors (u, v, w) where u,v,w are vectors of length 4.
    # M1: u=[1,0,1,0], v=[1,1,0,0], w=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # M4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,-1,0]
    # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,0]
    
    # I need to map these local indices to global indices.
    
    def add_strassen_block(a_indices, b_indices, c_indices, offset):
        # a_indices: [a, b, c, d] -> U indices
        # b_indices: [e, f, g, h] -> V indices
        # c_indices: [c11, c12, c21, c22] -> W indices
        
        # M1
        terms.append((a_indices[0], b_indices[0], c_indices[0], 1.0))
        terms.append((a_indices[0], b_indices[1], c_indices[1], 1.0))
        terms.append((a_indices[2], b_indices[0], c_indices[2], 1.0))
        terms.append((a_indices[2], b_indices[1], c_indices[3], 1.0))
        # Wait, the rank-1 tensors are not just products of single indices.
        # M1 = (a+d)(e+h) C11? No.
        # The decomposition is a sum of 7 rank-1 tensors.
        # T = sum_r U_r v_r W_r.
        # Each term is a product of a vector in U, V, W.
        # So I should add the vectors.
        
        pass

    # It is safer to explicitly define the 7 terms for each 2x2x2 block.
    
    # Block 1: A_11 * B_11[:, 0:2]
    # U indices: 0, 1, 3, 4
    # V indices: 0, 1, 4, 5
    # W indices: 0, 1, 3, 4
    
    # Block 2: A_11 * B_11[:, 2:4]
    # U indices: 0, 1, 3, 4
    # V indices: 2, 3, 6, 7
    # W indices: 6, 7, 9, 10
    
    # Block 3: A_12 * B_21 (Rank 1)
    # A_12: U indices 2, 5
    # B_21: V indices 8, 9, 10, 11
    # C_12: W indices 0, 1, 3, 4, 6, 7, 9, 10
    # Term: U[2,5] * V[8..11] * W[0,1,3,4,6,7,9,10]
    # This is a rank 1 term where U has 1s at 2,5; V has 1s at 8,9,10,11; W has 1s at 0,1,3,4,6,7,9,10.
    
    # Block 4: A_21 * B_11 (Rank 1)
    # A_21: U indices 6, 7
    # B_11: V indices 0,1,2,3,4,5,6,7
    # C_21: W indices 2, 5, 8, 11
    
    # Block 5: A_22 * B_21 (Rank 1)
    # A_22: U index 8
    # B_21: V indices 8,9,10,11
    # C_22: W indices 2, 5, 8, 11

    # Total terms: 7 + 7 + 1 + 1 + 1 = 17.
    
    # Let's populate the matrices.
    
    # Strassen terms definition
    # Local indices 0,1,2,3 correspond to 00, 01, 10, 11.
    strassen_terms = [
        # (u_vec, v_vec, w_vec)
        ([1,0,1,0], [1,1,0,0], [1,0,0,1]), # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]), # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]), # M3
        ([0,0,0,1], [0,0,1,-1], [0,0,1,0]), # M4
        ([1,1,0,0], [0,0,0,1], [0,0,-1,0]), # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]), # M6
        ([0,1,0,-1], [0,0,1,1], [0,1,0,0])  # M7
    ]
    
    def add_strassen(u_map, v_map, w_map, rank_offset):
        for u_v, v_v, w_v in strassen_terms:
            for i in range(4):
                if u_v[i] != 0:
                    U[u_map[i]][rank_offset] += u_v[i]
                if v_v[i] != 0:
                    V[v_map[i]][rank_offset] += v_v[i]
                if w_v[i] != 0:
                    W[w_map[i]][rank_offset] += w_v[i]

    # Block 1: A_11 * B_11[:, 0:2]
    u_map1 = [0, 1, 3, 4] # a,b,c,d
    v_map1 = [0, 1, 4, 5] # e,f,g,h
    w_map1 = [0, 1, 3, 4] # c11,c12,c21,c22
    add_strassen(u_map1, v_map1, w_map1, 0)
    
    # Block 2: A_11 * B_11[:, 2:4]
    u_map2 = [0, 1, 3, 4]
    v_map2 = [2, 3, 6, 7]
    w_map2 = [6, 7, 9, 10]
    add_strassen(u_map2, v_map2, w_map2, 7)
    
    # Block 3: A_12 * B_21
    # U indices: 2, 5
    # V indices: 8, 9, 10, 11
    # W indices: 0, 1, 3, 4, 6, 7, 9, 10
    rank_offset = 14
    for idx in [2, 5]:
        U[idx][rank_offset] = 1.0
    for idx in [8, 9, 10, 11]:
        V[idx][rank_offset] = 1.0
    for idx in [0, 1, 3, 4, 6, 7, 9, 10]:
        W[idx][rank_offset] = 1.0
        
    # Block 4: A_21 * B_11
    # U indices: 6, 7
    # V indices: 0, 1, 2, 3, 4, 5, 6, 7
    # W indices: 2, 5, 8, 11
    rank_offset = 15
    for idx in [6, 7]:
        U[idx][rank_offset] = 1.0
    for idx in [0, 1, 2, 3, 4, 5, 6, 7]:
        V[idx][rank_offset] = 1.0
    for idx in [2, 5, 8, 11]:
        W[idx][rank_offset] = 1.0
        
    # Block 5: A_22 * B_21
    # U index: 8
    # V indices: 8, 9, 10, 11
    # W indices: 2, 5, 8, 11
    rank_offset = 16
    U[8][rank_offset] = 1.0
    for idx in [8, 9, 10, 11]:
        V[idx][rank_offset] = 1.0
    for idx in [2, 5, 8, 11]:
        W[idx][rank_offset] = 1.0
        
    return {"U": U, "V": V, "W": W}
