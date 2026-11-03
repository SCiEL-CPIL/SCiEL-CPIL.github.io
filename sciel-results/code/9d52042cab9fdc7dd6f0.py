def construct():
    n, m, p = 2, 4, 8
    # U: (n*m, rank) = (8, rank), index [i*m+j]
    # V: (m*p, rank) = (32, rank), index [j*p+k]
    # W: (p*n, rank) = (16, rank), index [k*n+i]
    
    # We'll build lists of (U_vec, V_vec, W_vec) for each rank-1 term
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {A_index: coeff}, v_dict: {B_index: coeff}, w_dict: {C_index: coeff}"""
        terms.append((u_dict, v_dict, w_dict))
    
    # Block split:
    # A_L: i in {0,1}, j in {0,1} -> A indices: i*4+j for i in 0,1 and j in 0,1
    # A_R: i in {0,1}, j in {2,3} -> A indices: i*4+j for i in 0,1 and j in 2,3
    # B_T: j in {0,1}, k in {0..7} -> B indices: j*8+k
    # B_B: j in {2,3}, k in {0..7} -> B indices: j*8+k
    # C: i in {0,1}, k in {0..7} -> C indices: k*2+i
    
    def strassen_2x2_2x2(A_base, B_base, C_base, j_offset_A, j_offset_B, k_offset):
        """
        Apply Strassen to a 2x2 * 2x2 sub-product.
        A_base: base index for A row 0 (i.e., A[0, j_offset_A] is at A_base+0, A[1, j_offset_A] at A_base+1)
        Actually, let me redefine.
        
        A elements: A[i, j] at index i*m+j = i*4+j
        B elements: B[j, k] at index j*p+k = j*8+k
        C elements: C[i, k] at index k*n+i = k*2+i
        
        For a 2x2 A block at rows i in {i0, i0+1} and cols j in {j0, j0+1}:
          A[i0, j0] -> index i0*4+j0
          A[i0, j0+1] -> index i0*4+j0+1
          A[i0+1, j0] -> index (i0+1)*4+j0
          A[i0+1, j0+1] -> index (i0+1)*4+j0+1
        
        For a 2x2 B block at rows j in {j0, j0+1} and cols k in {k0, k0+1}:
          B[j0, k0] -> index j0*8+k0
          B[j0, k0+1] -> index j0*8+k0+1
          B[j0+1, k0] -> index (j0+1)*8+k0
          B[j0+1, k0+1] -> index (j0+1)*8+k0+1
        
        For a 2x2 C block at rows i in {i0, i0+1} and cols k in {k0, k0+1}:
          C[i0, k0] -> index k0*2+i0
          C[i0, k0+1] -> index (k0+1)*2+i0
          C[i0+1, k0] -> index k0*2+i0+1
          C[i0+1, k0+1] -> index (k0+1)*2+i0+1
        """
        # A indices
        a00 = i0 * 4 + j0
        a01 = i0 * 4 + j0 + 1
        a10 = (i0+1) * 4 + j0
        a11 = (i0+1) * 4 + j0 + 1
        
        # B indices
        b00 = j0 * 8 + k0
        b01 = j0 * 8 + k0 + 1
        b10 = (j0+1) * 8 + k0
        b11 = (j0+1) * 8 + k0 + 1
        
        # C indices
        c00 = k0 * 2 + i0
        c01 = (k0+1) * 2 + i0
        c10 = k0 * 2 + i0 + 1
        c11 = (k0+1) * 2 + i0 + 1
        
        # m1 = (a00+a11)(b00+b11) -> contributes to C11 (+1) and C22 (+1)
        # Wait, C11 is c11, C22 is... let me re-check.
        # C = [[C00, C01], [C10, C11]]
        # C00 = m1 - m4 + m6
        # C01 = m3 + m4
        # C10 = m2 + m5
        # C11 = m1 - m3 + m7
        
        # m1 = (a00+a11)(b00+b11)
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # m2 = (a10+a11)(b00)
        add_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1}
        )
        # m3 = a00(b01-b11)
        add_term(
            {a00: 1},
            {b01: 1, b11: -1},
            {c01: 1, c11: -1}
        )
        # m4 = a11(b10-b00)
        add_term(
            {a11: 1},
            {b10: 1, b00: -1},
            {c00: -1, c01: 1}
        )
        # m5 = (a00+a10)(b11+b10)
        add_term(
            {a00: 1, a10: 1},
            {b11: 1, b10: 1},
            {c10: 1}
        )
        # m6 = (a01-a11)(b00+b10)
        add_term(
            {a01: 1, a11: -1},
            {b00: 1, b10: 1},
            {c00: 1}
        )
        # m7 = (a01+a10)(b01-b11)
        add_term(
            {a01: 1, a10: 1},
            {b01: 1, b11: -1},
            {c11: 1}
        )
    
    # A_L: i in {0,1}, j in {0,1}; B_T: j in {0,1}, k in {0..7}
    # Split k into pairs: (0,1), (2,3), (4,5), (6,7)
    for k_pair in range(4):
        k0 = k_pair * 2
        strassen_2x2_2x2(i0=0, j0=0, k0=k0)
    
    # A_R: i in {0,1}, j in {2,3}; B_B: j in {2,3}, k in {0..7}
    for k_pair in range(4):
        k0 = k_pair * 2
        strassen_2x2_2x2(i0=0, j0=2, k0=k0)
    
    rank = len(terms)
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][col] = coeff
        for idx, coeff in v_dict.items():
            V[idx][col] = coeff
        for idx, coeff in w_dict.items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
