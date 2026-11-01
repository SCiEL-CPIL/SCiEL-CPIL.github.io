DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use 4 independent Strassen decompositions for 2x2 blocks
    # Rank per block is 7. Total rank is 28.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen multipliers for one 2x2 block
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # U vectors for one block (indices i*m+j)
    # a=0, b=1, c=2, d=3
    u_block = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ]
    
    # Strassen U factors for the 7 terms
    # Term 1: a+d -> u1 = [1, 0, 0, 1]
    # Term 2: c+d -> u2 = [0, 0, 1, 1]
    # Term 3: a   -> u3 = [1, 0, 0, 0]
    # Term 4: d   -> u4 = [0, 0, 0, 1]
    # Term 5: a+b -> u5 = [1, 1, 0, 0]
    # Term 6: c-a -> u6 = [-1, 0, 1, 0]
    # Term 7: b-d -> u7 = [0, 1, 0, -1]
    
    U_terms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V vectors for one block (indices j*p+k)
    # For a 2x2 block at column offset k_off, the columns are k_off, k_off+1
    # e = V[0, k_off], f = V[0, k_off+1]
    # g = V[1, k_off], h = V[1, k_off+1]
    
    # V factors for the 7 terms
    # Term 1: e+h -> v1 = [1, 1, 1, 1] (indices 0, 1, 4, 5 relative to block start? No, global)
    # Let's define relative indices for V in a 2x2 block:
    # 0: e, 1: f, 2: g, 3: h (if p=2)
    # But here p=8, so we need to map to global indices.
    
    # V_terms relative to block (0:e, 1:f, 2:g, 3:h)
    V_terms_rel = [
        [1, 1, 1, 1], # e+h
        [1, 0, 1, 0], # e
        [0, 1, 0, -1],# f-h
        [0, -1, 1, 0],# g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    # W vectors for one block (indices k*n+i)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W factors relative to block (0:C11, 1:C12, 2:C21, 3:C22)
    W_terms_rel = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Fill U, V, W
    for block in range(4):
        k_off = block * 2
        for t in range(7):
            col = block * 7 + t
            
            # U
            for i in range(2):
                for j in range(2):
                    U[i*m+j][col] = U_terms[t][i*m+j]
                    
            # V
            # Map relative V indices to global
            # Rel 0 -> e -> j=0, k=k_off
            # Rel 1 -> f -> j=0, k=k_off+1
            # Rel 2 -> g -> j=1, k=k_off
            # Rel 3 -> h -> j=1, k=k_off+1
            rel_indices = [
                (0, k_off),
                (0, k_off+1),
                (1, k_off),
                (1, k_off+1)
            ]
            for rel_idx, val in enumerate(V_terms_rel[t]):
                j, k = rel_indices[rel_idx]
                V[j*p+k][col] = val
                
            # W
            # Map relative W indices to global
            # Rel 0 -> C11 -> k=2*block, i=0
            # Rel 1 -> C12 -> k=2*block+1, i=0
            # Rel 2 -> C21 -> k=2*block, i=1
            # Rel 3 -> C22 -> k=2*block+1, i=1
            w_rel_indices = [
                (2*block, 0),
                (2*block+1, 0),
                (2*block, 1),
                (2*block+1, 1)
            ]
            for rel_idx, val in enumerate(W_terms_rel[t]):
                k_out, i_out = w_rel_indices[rel_idx]
                W[k_out*n+i_out][col] = val
                
    return {"U": U, "V": V, "W": W}
