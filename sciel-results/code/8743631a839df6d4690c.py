"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        for i in u_indices:
            U[i][col] += coeff
        for j in v_indices:
            V[j][col] += coeff
        for k in w_indices:
            W[k][col] += coeff
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # Split B into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:7]
    
    # Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B10+B11) -> C01
    # M4 = A11B11 -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B00) -> C00, C11
    
    def add_strassen(a_base, b_base, c_base):
        # a_base: (i, j) offset for A
        # b_base: (j, k) offset for B
        # c_base: (i, k) offset for C
        ai, aj = a_base
        bj, bk = b_base
        ci, ck = c_base
        
        # Indices in U, V, W
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        add_term([ai*m+aj, (ai+1)*m+(aj+1)], 
                 [bj*p+bk, (bj+1)*p+(bk+1)], 
                 [ck*n+ci, (ck+1)*n+(ci+1)])
        # M2
        add_term([(ai+1)*m+aj, (ai+1)*m+(aj+1)], 
                 [bj*p+bk], 
                 [(ck+1)*n+ci])
        # M3
        add_term([ai*m+aj], 
                 [(bj+1)*p+bk, (bj+1)*p+(bk+1)], 
                 [ck*n+(ci+1)])
        # M4
        add_term([(ai+1)*m+(aj+1)], 
                 [(bj+1)*p+(bk+1)], 
                 [ck*n+ci, (ck+1)*n+ci])
        # M5
        add_term([ai*m+aj, ai*m+(aj+1)], 
                 [(bj+1)*p+(bk+1)], 
                 [ck*n+ci, ck*n+(ci+1)])
        # M6
        add_term([(ai+1)*m+aj, ai*m+aj], 
                 [bj*p+bk, bj*p+(bk+1)], 
                 [(ck+1)*n+(ci+1)])
        # Note: M6 has coeff -1 for A00 part? 
        # (A10 - A00)(B00 + B01)
        # U: A10 (1), A00 (-1)
        # V: B00 (1), B01 (1)
        # W: C11 (1)
        # My add_term adds coeff to all. I need to handle signs.
        
        # Let's restart add_term to handle signs properly or just call multiple times.
        # For M6:
        # U: A10 (1), A00 (-1)
        # V: B00 (1), B01 (1)
        # W: C11 (1)
        
        # M7
        # (A01 - A11)(B10 + B00)
        # U: A01 (1), A11 (-1)
        # V: B10 (1), B00 (1)
        # W: C00 (1), C11 (1)
        
    # The add_term function above doesn't handle mixed signs well.
    # I will modify the logic to add individual contributions or pass a list of (index, coeff).
    
    # Let's rewrite the construction with a more flexible add_term.
    
    # Reset
    col = 0
    
    def add_term_flex(u, v, w):
        nonlocal col
        for idx, c in u:
            U[idx][col] += c
        for idx, c in v:
            V[idx][col] += c
        for idx, c in w:
            W[idx][col] += c
        col += 1

    # Strassen terms for block 1 (cols 0-1)
    # A base (0,0), B base (0,0), C base (0,0)
    # M1: (A00+A11)(B00+B11) -> C00, C11
    add_term_flex(
        [(0*m+0, 1), (1*m+1, 1)],
        [(0*p+0, 1), (1*p+1, 1)],
        [(0*n+0, 1), (1*n+1, 1)]
    )
    # M2: (A10+A11)B00 -> C10
    add_term_flex(
        [(1*m+0, 1), (1*m+1, 1)],
        [(0*p+0, 1)],
        [(0*n+1, 1)]
    )
    # M3: A00(B10+B11) -> C01
    add_term_flex(
        [(0*m+0, 1)],
        [(1*p+0, 1), (1*p+1, 1)],
        [(1*n+0, 1)]
    )
    # M4: A11B11 -> C00, C10
    add_term_flex(
        [(1*m+1, 1)],
        [(1*p+1, 1)],
        [(0*n+0, 1), (0*n+1, 1)]
    )
    # M5: (A00+A01)B11 -> C00, C01
    add_term_flex(
        [(0*m+0, 1), (0*m+1, 1)],
        [(1*p+1, 1)],
        [(0*n+0, 1), (1*n+0, 1)]
    )
    # M6: (A10-A00)(B00+B01) -> C11
    add_term_flex(
        [(1*m+0, 1), (0*m+0, -1)],
        [(0*p+0, 1), (0*p+1, 1)],
        [(1*n+1, 1)]
    )
    # M7: (A01-A11)(B10+B00) -> C00, C11
    add_term_flex(
        [(0*m+1, 1), (1*m+1, -1)],
        [(1*p+0, 1), (0*p+0, 1)],
        [(0*n+0, 1), (1*n+1, 1)]
    )
    
    # Repeat for B cols 2-3 (offset 2)
    # A base (0,0), B base (0,2), C base (0,2)
    add_term_flex(
        [(0*m+0, 1), (1*m+1, 1)],
        [(0*p+2, 1), (1*p+3, 1)],
        [(2*n+0, 1), (3*n+1, 1)]
    )
    add_term_flex(
        [(1*m+0, 1), (1*m+1, 1)],
        [(0*p+2, 1)],
        [(2*n+1, 1)]
    )
    add_term_flex(
        [(0*m+0, 1)],
        [(1*p+2, 1), (1*p+3, 1)],
        [(3*n+0, 1)]
    )
    add_term_flex(
        [(1*m+1, 1)],
        [(1*p+3, 1)],
        [(2*n+0, 1), (2*n+1, 1)]
    )
    add_term_flex(
        [(0*m+0, 1), (0*m+1, 1)],
        [(1*p+3, 1)],
        [(2*n+0, 1), (3*n+0, 1)]
    )
    add_term_flex(
        [(1*m+0, 1), (0*m+0, -1)],
        [(0*p+2, 1), (0*p+3, 1)],
        [(3*n+1, 1)]
    )
    add_term_flex(
        [(0*m+1, 1), (1*m+1, -1)],
        [(1*p+2, 1), (0*p+2, 1)],
        [(2*n+0, 1), (3*n+1, 1)]
    )
    
    # Repeat for B cols 4-5 (offset 4)
    add_term_flex(
        [(0*m+0, 1), (1*m+1, 1)],
        [(0*p+4, 1), (1*p+5, 1)],
        [(4*n+0, 1), (5*n+1, 1)]
    )
    add_term_flex(
        [(1*m+0, 1), (1*m+1, 1)],
        [(0*p+4, 1)],
        [(4*n+1, 1)]
    )
    add_term_flex(
        [(0*m+0, 1)],
        [(1*p+4, 1), (1*p+5, 1)],
        [(5*n+0, 1)]
    )
    add_term_flex(
        [(1*m+1, 1)],
        [(1*p+5, 1)],
        [(4*n+0, 1), (4*n+1, 1)]
    )
    add_term_flex(
        [(0*m+0, 1), (0*m+1, 1)],
        [(1*p+5, 1)],
        [(4*n+0, 1), (5*n+0, 1)]
    )
    add_term_flex(
        [(1*m+0, 1), (0*m+0, -1)],
        [(0*p+4, 1), (0*p+5, 1)],
        [(5*n+1, 1)]
    )
    add_term_flex(
        [(0*m+1, 1), (1*m+1, -1)],
        [(1*p+4, 1), (0*p+4, 1)],
        [(4*n+0, 1), (5*n+1, 1)]
    )
    
    # B col 6 (2x1 block)
    # A[0:2, 0:2] * B[0:2, 6]
    # C[0:2, 6]
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    add_term_flex([(0*m+0, 1)], [(0*p+6, 1)], [(6*n+0, 1)])
    add_term_flex([(0*m+1, 1)], [(1*p+6, 1)], [(6*n+0, 1)])
    add_term_flex([(1*m+0, 1)], [(0*p+6, 1)], [(6*n+1, 1)])
    add_term_flex([(1*m+1, 1)], [(1*p+6, 1)], [(6*n+1, 1)])
    
    # Block 2: A[2, 0:2] * B[0:2, 0:7]
    # A is 1x2, B is 2x7.
    # C[2, 0:7]
    # C2k = A20 B0k + A21 B1k
    for k in range(7):
        add_term_flex([(2*m+0, 1)], [(0*p+k, 1)], [(k*n+2, 1)])
        add_term_flex([(2*m+1, 1)], [(1*p+k, 1)], [(k*n+2, 1)])
        
    # Block 3: A[0:2, 2] * B[2, 0:7]
    # A is 2x1, B is 1x7.
    # C[0:2, k]
    # C0k = A02 B2k
    # C1k = A12 B2k
    for k in range(7):
        add_term_flex([(0*m+2, 1)], [(2*p+k, 1)], [(k*n+0, 1)])
        add_term_flex([(1*m+2, 1)], [(2*p+k, 1)], [(k*n+1, 1)])
        
    # Block 4: A[2, 2] * B[2, 0:7]
    # C[2, k] = A22 B2k
    for k in range(7):
        add_term_flex([(2*m+2, 1)], [(2*p+k, 1)], [(k*n+2, 1)])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
