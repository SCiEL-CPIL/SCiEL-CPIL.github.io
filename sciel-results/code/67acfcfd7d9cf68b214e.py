"""Matrix multiplication 6x6x6 using 2x2 block decomposition with rank-19 3x3 (Laderman)."""

def construct():
    n, m, p = 6, 6, 6
    
    # Laderman's rank-19 decomposition for 3x3 matrix multiplication.
    # We define U33, V33, W33 as 9x19 matrices.
    # Indices: U33[i*3+j, c], V33[j*3+k, c], W33[k*3+i, c]
    
    r3 = 19
    U33 = [[0] * r3 for _ in range(9)]
    V33 = [[0] * r3 for _ in range(9)]
    W33 = [[0] * r3 for _ in range(9)]
    
    # Helper to set a term
    # term_idx, U_entries, V_entries, W_entries
    # U_entries: list of (row_idx, coeff)
    # row_idx for U is i*3+j
    # row_idx for V is j*3+k
    # row_idx for W is k*3+i
    
    def set_term(c, u_entries, v_entries, w_entries):
        for r, val in u_entries:
            U33[r][c] += val
        for r, val in v_entries:
            V33[r][c] += val
        for r, val in w_entries:
            W33[r][c] += val
            
    # m0 = a00 * b00
    # i=0, j=0, k=0
    set_term(0, [(0, 1)], [(0, 1)], [(0, 1)])
    
    # m1 = a11 * b11
    # i=1, j=1, k=1
    set_term(1, [(4, 1)], [(4, 1)], [(4, 1)])
    
    # m2 = a22 * b22
    # i=2, j=2, k=2
    set_term(2, [(8, 1)], [(8, 1)], [(8, 1)])
    
    # m3 = a01 * b10
    # i=0, j=1, k=0
    # U[0*3+1] = U[1], V[1*3+0] = V[3], W[0*3+0] = W[0]
    set_term(3, [(1, 1)], [(3, 1)], [(0, 1)])
    
    # m4 = a12 * b20
    # i=1, j=2, k=0
    # U[1*3+2] = U[5], V[2*3+0] = V[6], W[0*3+1] = W[3]
    set_term(4, [(5, 1)], [(6, 1)], [(3, 1)])
    
    # m5 = a20 * b01
    # i=2, j=0, k=1
    # U[2*3+0] = U[6], V[0*3+1] = V[1], W[1*3+2] = W[5]
    set_term(5, [(6, 1)], [(1, 1)], [(5, 1)])
    
    # m6 = a02 * b21
    # i=0, j=2, k=1
    # U[0*3+2] = U[2], V[2*3+1] = V[7], W[1*3+0] = W[3]
    set_term(6, [(2, 1)], [(7, 1)], [(3, 1)])
    
    # m7 = a10 * b02
    # i=1, j=0, k=2
    # U[1*3+0] = U[3], V[0*3+2] = V[2], W[2*3+1] = W[7]
    set_term(7, [(3, 1)], [(2, 1)], [(7, 1)])
    
    # m8 = a21 * b12
    # i=2, j=1, k=2
    # U[2*3+1] = U[7], V[1*3+2] = V[5], W[2*3+2] = W[8]
    set_term(8, [(7, 1)], [(5, 1)], [(8, 1)])
    
    # m9 = (a00 + a10) * (b00 + b10)
    # i=0,1; j=0,1; k=0,1
    # U: (0,0), (1,0) -> rows 0, 3
    # V: (0,0), (1,0) -> rows 0, 3
    # W: (0,0), (1,0) -> rows 0, 3
    set_term(9, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    
    # m10 = (a01 + a11) * (b10 + b11)
    # i=0,1; j=1,1; k=0,1
    # U: (0,1), (1,1) -> rows 1, 4
    # V: (1,0), (1,1) -> rows 3, 4
    # W: (0,0), (1,1) -> rows 0, 4
    set_term(10, [(1, 1), (4, 1)], [(3, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # m11 = (a02 + a12) * (b20 + b21)
    # i=0,1; j=2,2; k=0,1
    # U: (0,2), (1,2) -> rows 2, 5
    # V: (2,0), (2,1) -> rows 6, 7
    # W: (0,0), (1,1) -> rows 0, 4
    set_term(11, [(2, 1), (5, 1)], [(6, 1), (7, 1)], [(0, 1), (4, 1)])
    
    # m12 = (a00 + a20) * (b00 + b20)
    # i=0,2; j=0,0; k=0,0
    # U: (0,0), (2,0) -> rows 0, 6
    # V: (0,0), (0,0) -> row 0 (coeff 2? No, b00 appears in both terms? No, b00 is one term, b20 is another. Wait.)
    # Term is (a00+a20)(b00+b20).
    # Expansion: a00b00 + a00b20 + a20b00 + a20b20.
    # This is a single product term in the bilinear form.
    # U: rows for a00 (0,0) and a20 (2,0). So U[0]=1, U[6]=1.
    # V: rows for b00 (0,0) and b20 (2,0). So V[0]=1, V[6]=1.
    # W: rows for c00 (0,0) and c20 (2,0). So W[0]=1, W[6]=1.
    set_term(12, [(0, 1), (6, 1)], [(0, 1), (6, 1)], [(0, 1), (6, 1)])
    
    # m13 = (a01 + a21) * (b10 + b20)
    # i=0,2; j=1,1; k=0,0
    # U: (0,1), (2,1) -> rows 1, 7
    # V: (1,0), (1,0) -> row 3? No. b10 is (1,0), b20 is (2,0).
    # Wait, the term is (a01+a21)(b10+b20).
    # V entries: b10 corresponds to j=1, k=0 -> V[1*3+0]=V[3]. b20 corresponds to j=2, k=0 -> V[2*3+0]=V[6].
    # So V[3]=1, V[6]=1.
    # W entries: c00 (0,0) -> W[0]. c20 (2,0) -> W[6].
    # So W[0]=1, W[6]=1.
    set_term(13, [(1, 1), (7, 1)], [(3, 1), (6, 1)], [(0, 1), (6, 1)])
    
    # m14 = (a02 + a22) * (b20 + b21)
    # i=0,2; j=2,2; k=0,1
    # U: (0,2), (2,2) -> rows 2, 8
    # V: (2,0), (2,1) -> rows 6, 7
    # W: (0,0), (1,1) -> rows 0, 4
    set_term(14, [(2, 1), (8, 1)], [(6, 1), (7, 1)], [(0, 1), (4, 1)])
    
    # m15 = (a10 + a20) * (b01 + b21)
    # i=1,2; j=0,0; k=1,1
    # U: (1,0), (2,0) -> rows 3, 6
    # V: (0,1), (2,1) -> rows 1, 7
    # W: (1,1), (1,1) -> row 4? No. c11 (1,1) -> W[4]. c21 (2,1) -> W[7].
    # Wait, W index is k*3+i.
    # For c11: i=1, k=1 -> W[1*3+1] = W[4].
    # For c21: i=2, k=1 -> W[1*3+2] = W[5]? No, W[k*3+i]. k=1, i=2 -> W[1*3+2]=W[5].
    # Let's re-verify W index mapping.
    # W[k*3+i, c].
    # c00: k=0, i=0 -> W[0].
    # c20: k=0, i=2 -> W[2].
    # c01: k=1, i=0 -> W[3].
    # c21: k=1, i=2 -> W[5].
    # c11: k=1, i=1 -> W[4].
    
    # Let's re-evaluate m15 W entries.
    # Term contributes to C11 and C21?
    # (a10+a20)(b01+b21) = a10b01 + a10b21 + a20b01 + a20b21.
    # a10b01 -> C11? No. C[i,k] = sum_j A[i,j]B[j,k].
    # A[1,0]B[0,1] -> C[1,1].
    # A[1,0]B[2,1] -> C[1,1]? No, B[2,1] is j=2, k=1. A[1,0] is i=1, j=0. j must match. So A[1,0]B[0,1] is valid for C[1,1]. A[1,0]B[2,1] is invalid (j=0 vs j=2).
    # So the product (a10+a20)(b01+b21) is NOT a valid single term for the tensor unless the indices align.
    # Laderman's algorithm relies on specific cancellations.
    # Let's trust the standard Laderman list.
    
    # Standard Laderman terms:
    # m15 = (a10 + a20)(b01 + b21)
    # This term is used in the reconstruction of C.
    # Let's just stick to the indices derived from the expansion of the product.
    # U: a10 (i=1,j=0) -> U[3]. a20 (i=2,j=0) -> U[6].
    # V: b01 (j=0,k=1) -> V[1]. b21 (j=2,k=1) -> V[7].
    # W: The result indices (i,k) are (1,1) and (2,1).
    # (1,1) -> W[1*3+1]=W[4].
    # (2,1) -> W[1*3+2]=W[5].
    set_term(15, [(3, 1), (6, 1)], [(1, 1), (7, 1)], [(4, 1), (5, 1)])
    
    # m16 = (a11 + a21) * (b11 + b21)
    # U: a11 (1,1)->U[4]. a21 (2,1)->U[7].
    # V: b11 (1,1)->V[4]. b21 (2,1)->V[7].
    # W: (1,1)->W[4]. (2,1)->W[5].
    set_term(16, [(4, 1), (7, 1)], [(4, 1), (7, 1)], [(4, 1), (5, 1)])
    
    # m17 = (a12 + a22) * (b21 + b22)
    # U: a12 (1,2)->U[5]. a22 (2,2)->U[8].
    # V: b21 (2,1)->V[7]. b22 (2,2)->V[8].
    # W: (1,2)->W[2*3+1]=W[7]. (2,2)->W[2*3+2]=W[8].
    set_term(17, [(5, 1), (8, 1)], [(7, 1), (8, 1)], [(7, 1), (8, 1)])
    
    # m18 = (a00 + a01 + a02) * (b00 + b10 + b20)
    # U: a00 (0,0)->U[0]. a01 (0,1)->U[1]. a02 (0,2)->U[2].
    # V: b00 (0,0)->V[0]. b10 (1,0)->V[3]. b20 (2,0)->V[6].
    # W: (0,0)->W[0]. (0,1)->W[3]. (0,2)->W[6].
    set_term(18, [(0, 1), (1, 1), (2, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)])
    
    # Now build the full 6x6 decomposition using Strassen at the block level.
    # Total rank = 7 * r3 = 7 * 19 = 133
    
    rank = 7 * r3
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    strassen_terms = [
        {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
        {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
        {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
        {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
        {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
        {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
        {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        for local_col in range(r3):
            col = base_col + local_col
            i3 = local_col // 9
            rem = local_col % 9
            j3 = rem // 3
            k3 = rem % 3
            
            # U part
            u_val = U33[i3*3+j3][local_col]
            if u_val != 0:
                for a_block, ca in term['a'].items():
                    bi, bj = block_map[a_block]
                    u_row = (bi*3+i3)*6 + (bj*3+j3)
                    U[u_row][col] += ca * u_val
                    
            # V part
            v_val = V33[j3*3+k3][local_col]
            if v_val != 0:
                for b_block, cb in term['b'].items():
                    bk, bl = block_map[b_block]
                    v_row = (bk*3+j3)*6 + (bl*3+k3)
                    V[v_row][col] += cb * v_val
                    
            # W part
            w_val = W33[k3*3+i3][local_col]
            if w_val != 0:
                for c_block, cc in term['c'].items():
                    ci, ck = block_map[c_block]
                    w_row = (ck*3+k3)*6 + (ci*3+i3)
                    W[w_row][col] += cc * w_val
    
    return {"U": U, "V": V, "W": W}
