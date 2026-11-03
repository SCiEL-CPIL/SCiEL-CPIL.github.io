"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(b-g)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U vectors (coeffs for a,b,c,d)
    U_strassen = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [0, 0, 1, -1],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # V vectors (coeffs for e,f,g,h)
    V_strassen = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, -1, 1, 0],# M4: b-g -> wait, b is from A. g is from B.
                      # M4 = d * (b-g). So V part is -g. Coeffs for e,f,g,h: [0,0,-1,0]
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    # Correction for M4:
    # M4 = d(b-g). U is d (index 3). V is -g (index 2).
    V_strassen[3] = [0, 0, -1, 0]
    
    # W vectors (coeffs for C00, C01, C10, C11)
    W_strassen = [
        [1, 0, 0, 0], # M1 in C00
        [0, 0, 0, 0], # M2 not in C00?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    
    # Let's define W properly for each term
    # Term 1 (M1): C00+=1, C11+=1
    # Term 2 (M2): C10+=1, C11-=1
    # Term 3 (M3): C01+=1, C11+=1
    # Term 4 (M4): C00+=1
    # Term 5 (M5): C00-=1, C01+=1
    # Term 6 (M6): C10+=1, C11+=1
    # Term 7 (M7): C00+=1
    
    W_strassen = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 0, 0], # M4: C00
        [-1, 1, 0, 0],# M5: C00, C01
        [0, 0, 1, 1], # M6: C10, C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    def add_strassen_block(offset_u, offset_v, offset_w, col_start):
        for t in range(7):
            col = col_start + t
            # U
            for i in range(2):
                for j in range(2):
                    idx_u = offset_u + i * 2 + j
                    U[idx_u][col] = U_strassen[t][i * 2 + j]
            # V
            for i in range(2):
                for j in range(2):
                    idx_v = offset_v + i * 2 + j
                    V[idx_v][col] = V_strassen[t][i * 2 + j]
            # W
            for i in range(2):
                for j in range(2):
                    idx_w = offset_w + i * 2 + j
                    W[idx_w][col] = W_strassen[t][i * 2 + j]

    def add_schoolbook_221_block(offset_u, offset_v, offset_w, col_start):
        # 2x2 A * 2x1 B
        # C0 = a*e + b*g
        # C1 = c*e + d*g
        # Terms: ae, bg, ce, dg
        coeffs = [
            (0, 0, 0, 0, 1, 0, 0, 1), # a*e -> C0
            (0, 1, 0, 0, 0, 0, 1, 1), # b*g -> C0
            (1, 0, 0, 0, 1, 0, 0, 2), # c*e -> C1
            (1, 1, 0, 0, 0, 0, 1, 2)  # d*g -> C1
        ]
        # U idx: 0:a, 1:b, 2:c, 3:d
        # V idx: 0:e, 1:f, 2:g, 3:h (f,h unused)
        # W idx: 0:C00, 1:C01, 2:C10, 3:C11 (Only C00, C10 used? No, C0, C1 are rows)
        # In W, indices are k*n+i.
        # Here k is fixed (the 1 column).
        # So W indices are 0 (k=0, i=0) and 1 (k=0, i=1)?
        # No, W is p*n.
        # The block is for a specific k.
        # So W offset points to W[k*n + 0] and W[k*n + 1].
        
        for t in range(4):
            col = col_start + t
            u_idx, v_idx, w_val = coeffs[t][0], coeffs[t][1], coeffs[t][7]
            # U
            U[offset_u + u_idx][col] = 1
            # V
            V[offset_v + v_idx][col] = 1
            # W
            # w_val is 1 for C0, 2 for C1?
            # C0 is i=0. C1 is i=1.
            # W index is k*n + i.
            # So for i=0, index is offset_w + 0.
            # For i=1, index is offset_w + 1.
            if w_val == 1:
                W[offset_w + 0][col] = 1
            else:
                W[offset_w + 1][col] = 1

    # Block 0: j in {0,1}
    # U offset: 0 (i*2 + j)
    # V offset: 0 (j*7 + k) -> j=0, so 0
    # W offset: 0 (k*2 + i)
    
    col = 0
    # k in {0,1}
    add_strassen_block(0, 0, 0, col)
    col += 7
    # k in {2,3}
    add_strassen_block(0, 14, 2, col) # V offset: 2*7=14? No.
    # V index is j*p + k.
    # For j=0, k=2,3. V indices are 2, 3.
    # Wait, V_strassen expects 4 indices e,f,g,h corresponding to B[0,0], B[0,1], B[1,0], B[1,1].
    # In the global V, for j=0, k=2,3:
    # B[0,2] is V[2]. B[0,3] is V[3].
    # B[1,2] is V[1*7+2] = V[9]. B[1,3] is V[10].
    # So the mapping is not contiguous in V for a 2x2 block if p=7.
    # V_strassen assumes contiguous e,f,g,h.
    # We need to map correctly.
    
    # Let's refine the block addition to take explicit index mappings.
    
    def add_strassen_block_explicit(u_map, v_map, w_map, col_start):
        # u_map: list of 4 global U indices for a,b,c,d
        # v_map: list of 4 global V indices for e,f,g,h
        # w_map: list of 4 global W indices for C00,C01,C10,C11
        
        for t in range(7):
            col = col_start + t
            for local in range(4):
                U[u_map[local]][col] = U_strassen[t][local]
                V[v_map[local]][col] = V_strassen[t][local]
                W[w_map[local]][col] = W_strassen[t][local]

    def add_schoolbook_221_explicit(u_map, v_map, w_map, col_start):
        # u_map: a,b,c,d
        # v_map: e,g (f,h ignored)
        # w_map: C0, C1
        
        for t in range(4):
            col = col_start + t
            if t == 0: # ae
                U[u_map[0]][col] = 1
                V[v_map[0]][col] = 1
                W[w_map[0]][col] = 1
            elif t == 1: # bg
                U[u_map[1]][col] = 1
                V[v_map[1]][col] = 1
                W[w_map[0]][col] = 1
            elif t == 2: # ce
                U[u_map[2]][col] = 1
                V[v_map[0]][col] = 1
                W[w_map[1]][col] = 1
            elif t == 3: # dg
                U[u_map[3]][col] = 1
                V[v_map[1]][col] = 1
                W[w_map[1]][col] = 1

    # Block 0 (j=0,1)
    # U indices:
    # a: 0*2+0=0, b: 0*2+1=1, c: 1*2+0=2, d: 1*2+1=3
    u_map_0 = [0, 1, 2, 3]
    
    # V indices for j=0:
    # e (B0,0): 0*7+0=0
    # f (B0,1): 0*7+1=1
    # g (B1,0): 1*7+0=7
    # h (B1,1): 1*7+1=8
    # For k=0,1: v_map = [0, 1, 7, 8]
    # For k=2,3: v_map = [2, 3, 9, 10]
    # For k=4,5: v_map = [4, 5, 11, 12]
    # For k=6:   v_map = [6, 13] (e, g)
    
    # W indices for j=0,1 (so C is 2x7):
    # C00: 0*2+0=0
    # C01: 0*2+1=1
    # C10: 1*2+0=2
    # C11: 1*2+1=3
    # For k=0,1: w_map = [0, 1, 2, 3]
    # For k=2,3: w_map = [4, 5, 6, 7]
    # For k=4,5: w_map = [8, 9, 10, 11]
    # For k=6:   w_map = [12, 13] (C0, C1)
    
    col = 0
    add_strassen_block_explicit(u_map_0, [0, 1, 7, 8], [0, 1, 2, 3], col)
    col += 7
    add_strassen_block_explicit(u_map_0, [2, 3, 9, 10], [4, 5, 6, 7], col)
    col += 7
    add_strassen_block_explicit(u_map_0, [4, 5, 11, 12], [8, 9, 10, 11], col)
    col += 7
    add_schoolbook_221_explicit(u_map_0, [6, 13], [12, 13], col)
    col += 4
    
    # Block 1 (j=2,3)
    # U indices:
    # a: 0*2+2=2, b: 0*2+3=3, c: 1*2+2=4, d: 1*2+3=5
    u_map_1 = [2, 3, 4, 5]
    
    # V indices for j=1:
    # e (B0,2): 0*7+2=2? No.
    # V index is j*p + k.
    # j=1 (second half of m=4? No, m=4, so j=0,1,2,3).
    # Block 1 is j=2,3.
    # So V indices:
    # e (B0,2): 2*7+0=14
    # f (B0,3): 2*7+1=15
    # g (B1,2): 3*7+0=21
    # h (B1,3): 3*7+1=22
    # For k=0,1: v_map = [14, 15, 21, 22]
    # For k=2,3: v_map = [16, 17, 23, 24]
    # For k=4,5: v_map = [18, 19, 25, 26]
    # For k=6:   v_map = [20, 27]
    
    # W indices are same as Block 0?
    # No, W is k*n + i.
    # i is 0,1. k is 0..6.
    # So W indices are the same for both blocks.
    
    add_strassen_block_explicit(u_map_1, [14, 15, 21, 22], [0, 1, 2, 3], col)
    col += 7
    add_strassen_block_explicit(u_map_1, [16, 17, 23, 24], [4, 5, 6, 7], col)
    col += 7
    add_strassen_block_explicit(u_map_1, [18, 19, 25, 26], [8, 9, 10, 11], col)
    col += 7
    add_schoolbook_221_explicit(u_map_1, [20, 27], [12, 13], col)
    col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
