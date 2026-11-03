# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    DEADLINE = 120.0
    
    # Strategy: Use a rank 12 decomposition for M_{2,4,2} for each of the 3 column blocks.
    # Total rank = 3 * 12 = 36.
    # M_{2,4,2} multiplies A (2x4) by B (4x2).
    # A = [[a, b, e, f], [c, d, g, h]]
    # B = [[x, y], [z, w], [u, v], [s, t]]
    # C = A B (2x2)
    # C_00 = ax + cz + eu + gs
    # C_01 = ay + cw + ev + gt
    # C_10 = bx + dz + fu + hs
    # C_11 = by + dw + fv + ht
    
    # Rank 12 decomposition terms (u, v, w)
    # u: linear combo of A entries
    # v: linear combo of B entries
    # w: linear combo of C entries (output)
    
    # Indices for A:
    # 0: a, 1: b, 2: e, 3: f
    # 4: c, 5: d, 6: g, 7: h
    # Indices for B:
    # 0: x, 1: y
    # 2: z, 3: w
    # 4: u, 5: v
    # 6: s, 7: t
    # Indices for C:
    # 0: C_00, 1: C_01
    # 2: C_10, 3: C_11
    
    # Terms:
    # 1. (a+e)(x+u) -> w: C_00
    # 2. (c+g)(z+s) -> w: C_00
    # 3. (a+b+e+f)(x+y+u+v) -> w: C_00 + C_01 + C_10 + C_11 ? No, w is a vector.
    # The output w is a linear combination of C entries.
    # Actually, the tensor decomposition is sum u_i v_i w_i^T.
    # w_i is a 4-vector (flattened 2x2 C).
    
    # Let's define the 12 terms explicitly.
    # A indices: 0..7
    # B indices: 0..7
    # C indices: 0..3
    
    terms = [
        # 1
        (
            [1, 0, 1, 0, 0, 0, 0, 0], # a+e
            [1, 0, 0, 0, 1, 0, 0, 0], # x+u
            [1, 0, 0, 0]              # C_00
        ),
        # 2
        (
            [0, 0, 0, 0, 1, 0, 1, 0], # c+g
            [0, 0, 1, 0, 0, 0, 1, 0], # z+s
            [1, 0, 0, 0]              # C_00
        ),
        # 3
        (
            [1, 1, 1, 1, 0, 0, 0, 0], # a+b+e+f
            [1, 1, 0, 0, 1, 1, 0, 0], # x+y+u+v
            [1, 1, 1, 1]              # C_00+C_01+C_10+C_11
        ),
        # 4
        (
            [0, 0, 0, 0, 1, 1, 1, 1], # c+d+g+h
            [0, 0, 1, 1, 0, 0, 1, 1], # z+w+s+t
            [1, 1, 1, 1]              # C_00+C_01+C_10+C_11
        ),
        # 5
        (
            [1, 0, 1, 0, -1, 0, -1, 0], # a-c+e-g
            [1, 0, -1, 0, 1, 0, -1, 0], # x-z+u-s
            [1, 0, 0, -1]               # C_00 - C_11
        ),
        # 6
        (
            [0, 1, 0, 1, 0, -1, 0, -1], # b-d+f-h
            [0, 1, 0, -1, 0, 1, 0, -1], # y-w+v-t
            [0, 1, -1, 0]               # C_01 - C_10
        ),
        # 7
        (
            [1, 0, 1, 0, 0, 0, 0, 0], # a+e
            [0, 1, 0, 0, 0, 1, 0, 0], # y+v
            [0, 1, 0, 0]              # C_01
        ),
        # 8
        (
            [0, 1, 0, 1, 0, 0, 0, 0], # b+f
            [1, 0, 0, 0, 1, 0, 0, 0], # x+u
            [0, 0, 1, 0]              # C_10
        ),
        # 9
        (
            [0, 0, 0, 0, 1, 0, 1, 0], # c+g
            [0, 0, 0, 1, 0, 0, 0, 1], # w+t
            [0, 1, 0, 0]              # C_01
        ),
        # 10
        (
            [0, 0, 0, 0, 0, 1, 0, 1], # d+h
            [0, 0, 1, 0, 0, 0, 1, 0], # z+s
            [0, 0, 1, 0]              # C_10
        ),
        # 11
        (
            [1, 1, 1, 1, 0, 0, 0, 0], # a+b+e+f
            [0, 1, 0, -1, 0, 1, 0, -1], # y-w+v-t
            [0, 1, -1, 0]              # C_01 - C_10
        ),
        # 12
        (
            [0, 0, 0, 0, 1, 1, 1, 1], # c+d+g+h
            [0, 0, -1, 1, 0, 0, 1, -1], # -z+w+s-t
            [0, -1, 1, 0]              # -C_01 + C_10
        ),
    ]
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_idx = 0
    for c in range(3):  # 3 column blocks
        for t in range(12):
            u_vec, v_vec, w_vec = terms[t]
            
            # Map A indices to U indices
            # A is 2x4.
            # A[0,0]=a, A[0,1]=b, A[0,2]=e, A[0,3]=f
            # A[1,0]=c, A[1,1]=d, A[1,2]=g, A[1,3]=h
            # U index = row * 4 + col
            # 0: a (0,0) -> 0
            # 1: b (0,1) -> 1
            # 2: e (0,2) -> 2
            # 3: f (0,3) -> 3
            # 4: c (1,0) -> 4
            # 5: d (1,1) -> 5
            # 6: g (1,2) -> 6
            # 7: h (1,3) -> 7
            
            for i, coeff in enumerate(u_vec):
                if coeff != 0:
                    U[i, col_idx] = coeff
            
            # Map B indices to V indices
            # B is 4x2.
            # B[0,0]=x, B[0,1]=y
            # B[1,0]=z, B[1,1]=w
            # B[2,0]=u, B[2,1]=v
            # B[3,0]=s, B[3,1]=t
            # V index = row * 2 + col
            # 0: x (0,0) -> 0
            # 1: y (0,1) -> 1
            # 2: z (1,0) -> 2
            # 3: w (1,1) -> 3
            # 4: u (2,0) -> 4
            # 5: v (2,1) -> 5
            # 6: s (3,0) -> 6
            # 7: t (3,1) -> 7
            
            # The block c corresponds to columns 2c and 2c+1 of the full B (4x6).
            # So B_block is B[:, 2c:2c+2].
            # V index for full B: row * 6 + (2c + col)
            
            for i, coeff in enumerate(v_vec):
                if coeff != 0:
                    # i is index in 4x2 block.
                    # row = i // 2, col = i % 2
                    row = i // 2
                    col = i % 2
                    v_idx = row * 6 + (2*c + col)
                    V[v_idx, col_idx] = coeff
            
            # Map C indices to W indices
            # C is 2x2 block.
            # C[0,0]=C_00, C[0,1]=C_01
            # C[1,0]=C_10, C[1,1]=C_11
            # W index = (col * 2 + row)? No, W[k*n+i, r] where k is col of C, i is row of C.
            # W index = (col * n) + row. n=2.
            # 0: C_00 (0,0) -> 0
            # 1: C_01 (0,1) -> 2*0 + 1? No.
            # W[k*n+i, r]. k is column index of C (0..1), i is row index of C (0..1).
            # C_00: k=0, i=0 -> 0
            # C_01: k=1, i=0 -> 2
            # C_10: k=0, i=1 -> 1
            # C_11: k=1, i=1 -> 3
            # Wait, C is 2x2.
            # C_00 is at (0,0). C_01 is at (0,1).
            # k is column. i is row.
            # C_00: k=0, i=0 -> 0*2+0=0.
            # C_01: k=1, i=0 -> 1*2+0=2.
            # C_10: k=0, i=1 -> 0*2+1=1.
            # C_11: k=1, i=1 -> 1*2+1=3.
            
            # The block c contributes to columns 2c and 2c+1 of the full C (6x2).
            # So C_block is C[:, 2c:2c+2].
            # W index for full C: ( (2c + k) * n ) + i.
            
            for i, coeff in enumerate(w_vec):
                if coeff != 0:
                    # i is index in 2x2 block.
                    # 0: C_00 -> k=0, i_row=0
                    # 1: C_01 -> k=1, i_row=0
                    # 2: C_10 -> k=0, i_row=1
                    # 3: C_11 -> k=1, i_row=1
                    if i == 0:
                        k, i_row = 0, 0
                    elif i == 1:
                        k, i_row = 1, 0
                    elif i == 2:
                        k, i_row = 0, 1
                    else:
                        k, i_row = 1, 1
                    
                    w_idx = (2*c + k) * n + i_row
                    W[w_idx, col_idx] = coeff
            
            col_idx += 1
            if time.time() - start_time > DEADLINE - 5.0:
                break
        if time.time() - start_time > DEADLINE - 5.0:
            break
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
