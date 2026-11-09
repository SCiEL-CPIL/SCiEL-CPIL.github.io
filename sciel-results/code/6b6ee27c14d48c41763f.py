import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define blocks for m (5) and p (5)
    # J1: 0,1 (size 2), J2: 2,3 (size 2), J3: 4 (size 1)
    # K1: 0,1 (size 2), K2: 2,3 (size 2), K3: 4 (size 1)
    
    j_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    k_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    
    col = 0
    
    # Helper to add a decomposition term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        # u_vec length n*m_loc, v_vec length m_loc*p_loc, w_vec length p_loc*n
        # But we need to map to global indices.
        pass

    # We will iterate blocks
    for jb_idx, jb in enumerate(j_blocks):
        for kb_idx, kb in enumerate(k_blocks):
            m_loc = len(jb)
            p_loc = len(kb)
            
            # Determine local rank and decomposition
            if m_loc == 2 and p_loc == 2:
                # Strassen 2x2x2
                # Local indices: i in 0..1, j in 0..1, k in 0..1
                # Global j = jb[0] + j_local
                # Global k = kb[0] + k_local
                
                # Strassen matrices
                # M1 = (A11+A22)(B11+B22)
                # M2 = (A21+A22)B11
                # M3 = A11(B21-B22)
                # M4 = A22(B12-B11)
                # M5 = (A11+A12)B22
                # M6 = (A21-A11)(B11+B12)
                # M7 = (A12-A22)(B21+B22)
                
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                terms = [
                    # (u_coeffs, v_coeffs, w_coeffs)
                    # u: A00, A01, A10, A11
                    # v: B00, B01, B10, B11
                    # w: C00, C01, C10, C11 (index k*n+i)
                    
                    # M1
                    ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0]),
                    # M4
                    ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 0]),
                    # -M5
                    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
                    # M7
                    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
                    
                    # M3
                    ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0]),
                    # M5
                    ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
                    
                    # M2
                    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
                    # M4
                    ([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0]),
                    
                    # -M2
                    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, -1]),
                    # M3
                    ([1, 0, 0, 0], [0, -1, 1, 0], [0, 0, 0, 1]),
                    # M6
                    ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
                    # M1
                    ([1, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1])
                ]
                
                # Wait, the above terms are not exactly the Strassen decomposition in terms of single products.
                # Strassen defines 7 products. Then sums them.
                # We need to represent the tensor as sum of rank 1 tensors.
                # Each Strassen product M_r is a rank 1 tensor in the A,B space, but it contributes to multiple C entries.
                # So M_r corresponds to a term: U_r (vec) * V_r (vec) * W_r (vec).
                # Where W_r has non-zeros in the C positions affected by M_r.
                
                # Let's redefine the 7 Strassen terms properly.
                # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
                # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
                # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
                # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
                # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
                # M6 = (A10-A00)(B00+B01) -> C11 += M6
                # M7 = (A01-A11)(B10+B11) -> C00 += M7
                
                strassen_terms = [
                    # M1
                    ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),
                    # M2
                    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                    # M3
                    ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),
                    # M4
                    ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),
                    # M5
                    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
                    # M6
                    ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
                    # M7
                    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
                ]
                
                for u_loc, v_loc, w_loc in strassen_terms:
                    # Map local indices to global
                    # u_loc indices: 0:A00, 1:A01, 2:A10, 3:A11
                    # Global U index: i*m + j
                    # i=0, j=jb[0]+0 -> jb[0]
                    # i=0, j=jb[0]+1 -> jb[0]+1
                    # i=1, j=jb[0]+0 -> m + jb[0]
                    # i=1, j=jb[0]+1 -> m + jb[0]+1
                    
                    u_global = [0] * (n * m_loc)
                    u_global[0] = u_loc[0]
                    u_global[1] = u_loc[1]
                    u_global[2] = u_loc[2]
                    u_global[3] = u_loc[3]
                    
                    # v_loc indices: 0:B00, 1:B01, 2:B10, 3:B11
                    # Global V index: j*p + k
                    # j=jb[0]+0, k=kb[0]+0
                    # j=jb[0]+0, k=kb[0]+1
                    # j=jb[0]+1, k=kb[0]+0
                    # j=jb[0]+1, k=kb[0]+1
                    
                    v_global = [0] * (m_loc * p_loc)
                    v_global[0] = v_loc[0]
                    v_global[1] = v_loc[1]
                    v_global[2] = v_loc[2]
                    v_global[3] = v_loc[3]
                    
                    # w_loc indices: 0:C00, 1:C01, 2:C10, 3:C11
                    # Global W index: k*n + i
                    # k=kb[0]+0, i=0
                    # k=kb[0]+1, i=0
                    # k=kb[0]+0, i=1
                    # k=kb[0]+1, i=1
                    
                    w_global = [0] * (p_loc * n)
                    w_global[0] = w_loc[0]
                    w_global[1] = w_loc[1]
                    w_global[2] = w_loc[2]
                    w_global[3] = w_loc[3]
                    
                    # Add to global matrices
                    for i in range(n * m_loc):
                        U[jb_idx * 0 + i][col] += u_global[i] # Note: U index is i*m + j. Here i is row of A.
                        # Wait, U shape is (n*m, rank). Index is i*m + j.
                        # In local block, j is 0..m_loc-1. Global j is jb[0]+j.
                        # So global index is i*m + (jb[0]+j).
                        # My u_global is indexed by local (i, j).
                        # I need to map local (i, j) to global i*m + j_global.
                        
                    # Let's do the mapping carefully in the loop below.
                    pass

            # This is getting complex. I will write a helper function to add a block decomposition.
            
    # Re-implementing with helper
    def add_block_decomposition(jb, kb, terms):
        nonlocal col
        m_loc = len(jb)
        p_loc = len(kb)
        
        for u_loc, v_loc, w_loc in terms:
            # u_loc is list of length n*m_loc? No, length n*m_loc is not right.
            # u_loc should correspond to A entries in this block.
            # A is n x m. Block is n x m_loc.
            # So u_loc length n*m_loc.
            # v_loc length m_loc*p_loc.
            # w_loc length p_loc*n.
            
            # Map u
            for i in range(n):
                for j_l in range(m_loc):
                    j_g = jb[0] + j_l
                    idx_u = i * m + j_g
                    U[idx_u][col] += u_loc[i * m_loc + j_l]
            
            # Map v
            for j_l in range(m_loc):
                for k_l in range(p_loc):
                    j_g = jb[0] + j_l
                    k_g = kb[0] + k_l
                    idx_v = j_g * p + k_g
                    V[idx_v][col] += v_loc[j_l * p_loc + k_l]
            
            # Map w
            for k_l in range(p_loc):
                for i in range(n):
                    k_g = kb[0] + k_l
                    idx_w = k_g * n + i
                    W[idx_w][col] += w_loc[k_l * n + i]
            
            col += 1

    # Define terms for each block type
    
    # Type 2x2x2 (Strassen)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Type 2x2x1 (Rank 4)
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # u: A00, A01, A10, A11
    # v: B00, B10
    # w: C0, C1
    rank4_221_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]
    
    # Type 1x2x2 (Rank 1)
    # C_i,k = A_i,0 B_0,k
    # u: A0, A1
    # v: B0
    # w: C00, C01, C10, C11
    rank1_122_terms = [
        ([1, 1], [1], [1, 1, 1, 1])
    ]
    
    # Type 1x1x1 (Rank 1)
    rank1_111_terms = [
        ([1], [1], [1])
    ]

    for jb in j_blocks:
        for kb in k_blocks:
            m_loc = len(jb)
            p_loc = len(kb)
            
            if m_loc == 2 and p_loc == 2:
                add_block_decomposition(jb, kb, strassen_terms)
            elif m_loc == 2 and p_loc == 1:
                add_block_decomposition(jb, kb, rank4_221_terms)
            elif m_loc == 1 and p_loc == 2:
                add_block_decomposition(jb, kb, rank1_122_terms)
            elif m_loc == 1 and p_loc == 1:
                add_block_decomposition(jb, kb, rank1_111_terms)
            
    return {"U": U, "V": V, "W": W}

