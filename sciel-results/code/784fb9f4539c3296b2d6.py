import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4x1 multiplication using Schoolbook (Rank 16)
    # Total Rank = 65
    
    rank_total = 65
    
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    # --- Part 1: 4x4x4 Strassen ---
    # Strassen rank-7 decomposition for 2x2x2
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas:
    # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    # m2 = (c+d)e    -> C10 += 1, C11 -= 1
    # m3 = a(f-h)    -> C01 += 1, C11 += 1
    # m4 = d(g-e)    -> C00 += 1, C10 += 1
    # m5 = (a+b)h    -> C00 -= 1, C01 += 1
    # m6 = (c-a)(e+f)-> C11 += 1
    # m7 = (b-d)(g+h)-> C00 += 1
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    def add_term(U, V, W, idx, u_idx, v_idx, w_idx, val=1):
        U[u_idx, idx] += val
        V[v_idx, idx] += val
        W[w_idx, idx] += val

    # m1: (a+d)(e+h)
    # A: a(0), d(3). B: e(0), h(3). C: c00(0), c11(3)
    add_term(U2, V2, W2, 0, 0, 0, 0, 1)
    add_term(U2, V2, W2, 0, 3, 0, 3, 1)
    add_term(U2, V2, W2, 0, 0, 3, 0, 1)
    add_term(U2, V2, W2, 0, 3, 3, 3, 1)
    
    # m2: (c+d)e
    # A: c(2), d(3). B: e(0). C: c10(2), c11(3)
    add_term(U2, V2, W2, 1, 2, 0, 2, 1)
    add_term(U2, V2, W2, 1, 3, 0, 2, 1)
    add_term(U2, V2, W2, 1, 2, 0, 3, -1)
    add_term(U2, V2, W2, 1, 3, 0, 3, -1)
    
    # m3: a(f-h)
    # A: a(0). B: f(1), h(3). C: c01(1), c11(3)
    add_term(U2, V2, W2, 2, 0, 1, 1, 1)
    add_term(U2, V2, W2, 2, 0, 3, 1, -1)
    add_term(U2, V2, W2, 2, 0, 1, 3, 1)
    add_term(U2, V2, W2, 2, 0, 3, 3, -1)
    
    # m4: d(g-e)
    # A: d(3). B: g(2), e(0). C: c00(0), c10(2)
    add_term(U2, V2, W2, 3, 3, 2, 0, 1)
    add_term(U2, V2, W2, 3, 3, 0, 0, -1)
    add_term(U2, V2, W2, 3, 3, 2, 2, 1)
    add_term(U2, V2, W2, 3, 3, 0, 2, -1)
    
    # m5: (a+b)h
    # A: a(0), b(1). B: h(3). C: c00(0), c01(1)
    add_term(U2, V2, W2, 4, 0, 3, 0, -1)
    add_term(U2, V2, W2, 4, 1, 3, 0, -1)
    add_term(U2, V2, W2, 4, 0, 3, 1, 1)
    add_term(U2, V2, W2, 4, 1, 3, 1, 1)
    
    # m6: (c-a)(e+f)
    # A: c(2), a(0). B: e(0), f(1). C: c11(3)
    add_term(U2, V2, W2, 5, 2, 0, 3, 1)
    add_term(U2, V2, W2, 5, 0, 0, 3, -1)
    add_term(U2, V2, W2, 5, 2, 1, 3, 1)
    add_term(U2, V2, W2, 5, 0, 1, 3, -1)
    
    # m7: (b-d)(g+h)
    # A: b(1), d(3). B: g(2), h(3). C: c00(0)
    add_term(U2, V2, W2, 6, 1, 2, 0, 1)
    add_term(U2, V2, W2, 6, 3, 2, 0, -1)
    add_term(U2, V2, W2, 6, 1, 3, 0, 1)
    add_term(U2, V2, W2, 6, 3, 3, 0, -1)

    # Strassen 4x4x4
    # Blocks of A: A00, A01, A10, A11 (indices 0,1,2,3 in 2x2 grid)
    # Blocks of B: B00, B01, B10, B11
    # C = A B
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to global U, V, W.
    # Global U index: i*m + j.
    # For block (bi, bj), local (li, lj), global index is (bi*2 + li)*4 + (bj*2 + lj).
    # Global V index: j*p + k. Here p=4 for this part.
    # For block (bj, bk), local (lj, lk), global index is (bj*2 + lj)*4 + (bk*2 + lk).
    # Global W index: k*n + i. Here n=4.
    # For block (bk, bi), local (lk, li), global index is (bk*2 + lk)*4 + (bi*2 + li).
    
    # Define Strassen block definitions:
    # Each block is defined by:
    # P_coeffs: dict {block_idx: coeff} for A
    # Q_coeffs: dict {block_idx: coeff} for B
    # C_blocks: dict {block_idx: coeff} for C
    
    strassen_blocks = [
        # M1
        {'P': {0: 1, 3: 1}, 'Q': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        # M2
        {'P': {2: 1, 3: 1}, 'Q': {0: 1}, 'C': {2: 1, 3: -1}},
        # M3
        {'P': {0: 1}, 'Q': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        # M4
        {'P': {3: 1}, 'Q': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        # M5
        {'P': {0: 1, 1: 1}, 'Q': {3: 1}, 'C': {0: -1, 1: 1}},
        # M6
        {'P': {2: 1, 0: -1}, 'Q': {0: 1, 1: 1}, 'C': {3: 1}},
        # M7
        {'P': {1: 1, 3: -1}, 'Q': {2: 1, 3: 1}, 'C': {0: 1}},
    ]
    
    current_rank = 0
    for s_idx, block in enumerate(strassen_blocks):
        for c in range(7):
            # For each local rank component c
            # We need to add the contribution of U2[:, c], V2[:, c], W2[:, c]
            # scaled by the coefficients of P, Q, C blocks.
            
            # U part
            for p_block, p_coeff in block['P'].items():
                # p_block is 0..3. bi = p_block // 2, bj = p_block % 2
                bi = p_block // 2
                bj = p_block % 2
                for li in range(2):
                    for lj in range(2):
                        u_local_idx = li * 2 + lj
                        val = U2[u_local_idx, c] * p_coeff
                        if val != 0:
                            # Global U index
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            u_global_idx = gi * 4 + gj
                            U[u_global_idx, current_rank] += val
            
            # V part
            for q_block, q_coeff in block['Q'].items():
                bj = q_block // 2
                bk = q_block % 2
                for lj in range(2):
                    for lk in range(2):
                        v_local_idx = lj * 2 + lk
                        val = V2[v_local_idx, c] * q_coeff
                        if val != 0:
                            # Global V index (p=4 for this part)
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            v_global_idx = gj * 4 + gk
                            V[v_global_idx, current_rank] += val
            
            # W part
            for c_block, c_coeff in block['C'].items():
                bk = c_block // 2
                bi = c_block % 2
                for lk in range(2):
                    for li in range(2):
                        w_local_idx = lk * 2 + li
                        val = W2[w_local_idx, c] * c_coeff
                        if val != 0:
                            # Global W index (n=4)
                            gk = bk * 2 + lk
                            gi = bi * 2 + li
                            w_global_idx = gk * 4 + gi
                            W[w_global_idx, current_rank] += val
            
            current_rank += 1
    
    # --- Part 2: 4x4x1 Schoolbook ---
    # A is 4x4. B1 is 4x1.
    # U index: i*4 + j
    # V index: j*1 + 0 = j
    # W index: 0*4 + i = i
    
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j
            w_idx = i
            U[u_idx, current_rank] = 1
            V[v_idx, current_rank] = 1
            W[w_idx, current_rank] = 1
            current_rank += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
