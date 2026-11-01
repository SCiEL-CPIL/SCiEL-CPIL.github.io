"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] = u_val
        V[v_idx][c] = v_val
        W[w_idx][c] = w_val
        c += 1

    # Part 1: A_L B_L (2x1 x 1x5) -> Rank 10
    # A_L is col 0 of A. B_L is row 0 of B.
    # A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, 1, v_idx, 1, w_idx, 1)

    # Part 2: A_R B_R (2x2 x 2x5)
    # A_R is cols 1,2 of A. B_R is rows 1,2 of B.
    # A_R indices in U: i*m + 1, i*m + 2
    # B_R indices in V: 1*p + k, 2*p + k
    
    # Split B_R into B_R1 (2x2) and B_R2 (2x3)
    # B_R1: cols 0,1 of B_R (which are cols 1,2 of B). Indices k=1,2 in original B?
    # Original B is 3x5.
    # B_R is rows 1,2. So B[1, :] and B[2, :].
    # B_R1 uses B[1, 0..1] and B[2, 0..1]?
    # Let's stick to local indices for the 2x2 block.
    # A_R is 2x2. Let's call its elements A_00, A_01, A_10, A_11 corresponding to original A[i, 1], A[i, 2].
    # B_R1 is 2x2. Elements B_00, B_01, B_10, B_11 corresponding to original B[1, j], B[2, j] for j=0,1.
    
    # 2.1: A_R B_R1 (2x2 x 2x2) Strassen Rank 7
    # Mapping to original indices:
    # A_00 -> U[i=0, j=1] -> idx 0*3+1 = 1
    # A_01 -> U[i=0, j=2] -> idx 0*3+2 = 2
    # A_10 -> U[i=1, j=1] -> idx 1*3+1 = 4
    # A_11 -> U[i=1, j=2] -> idx 1*3+2 = 5
    
    # B_00 -> V[i=1, j=0] -> idx 1*5+0 = 5
    # B_01 -> V[i=1, j=1] -> idx 1*5+1 = 6
    # B_10 -> V[i=2, j=0] -> idx 2*5+0 = 10
    # B_11 -> V[i=2, j=1] -> idx 2*5+1 = 11
    
    # W indices for this 2x2 block result (C is 2x5, we are filling cols 0,1 of C? No, B_R1 corresponds to B cols 0,1)
    # Result C_00 (i=0, k=0) -> W[0*2+0]=0
    # Result C_01 (i=0, k=1) -> W[1*2+0]=2
    # Result C_10 (i=1, k=0) -> W[0*2+1]=1
    # Result C_11 (i=1, k=1) -> W[1*2+1]=3
    
    # Strassen M1: (A_00+A_11)(B_00+B_11) -> C_00(1), C_01(1), C_10(1), C_11(1)
    # U: 1@1, 1@5
    # V: 1@5, 1@11
    # W: 1@0, 1@2, 1@1, 1@3
    add_term(1, 1, 5, 1, 0, 1)
    add_term(1, 0, 5, 0, 2, 1) # Wait, add_term handles one (u,v,w) triple per call?
    # My add_term adds to a single column c.
    # I need to set multiple entries in the same column c.
    
    # Refined add_term for Strassen
    def add_strassen_term(u_offsets, v_offsets, w_offsets):
        nonlocal c
        for u_off, u_val in u_offsets:
            U[u_off][c] = u_val
        for v_off, v_val in v_offsets:
            V[v_off][c] = v_val
        for w_off, w_val in w_offsets:
            W[w_off][c] = w_val
        c += 1

    # Re-initialize c if needed? No, c is 10 now.
    
    # M1: (A00+A11)(B00+B11)
    add_strassen_term(
        [(1, 1), (5, 1)], 
        [(5, 1), (11, 1)], 
        [(0, 1), (2, 1), (1, 1), (3, 1)]
    )
    
    # M2: (A10+A11)B00
    add_strassen_term(
        [(4, 1), (5, 1)], 
        [(5, 1)], 
        [(1, 1), (3, 1)] # C10, C11
    )
    
    # M3: A00(B01-B11)
    add_strassen_term(
        [(1, 1)], 
        [(6, 1), (11, -1)], 
        [(2, 1), (3, 1)] # C01, C11
    )
    
    # M4: A11(B10-B00)
    add_strassen_term(
        [(5, 1)], 
        [(10, 1), (5, -1)], 
        [(1, 1), (3, 1)] # C10, C11
    )
    
    # M5: (A00+A01)B11
    add_strassen_term(
        [(1, 1), (2, 1)], 
        [(11, 1)], 
        [(0, 1), (2, 1)] # C00, C01
    )
    
    # M6: (A10-A00)(B00+B01)
    add_strassen_term(
        [(4, 1), (1, -1)], 
        [(5, 1), (6, 1)], 
        [(3, 1)] # C11
    )
    
    # M7: (A01-A11)(B10+B11)
    add_strassen_term(
        [(2, 1), (5, -1)], 
        [(10, 1), (11, 1)], 
        [(0, 1)] # C00
    )
    
    # Part 2.2: A_R B_R2 (2x2 x 2x3)
    # B_R2 is 2x3. Cols 2,3,4 of original B.
    # Split into B_R2_1 (2x1) col 2 and B_R2_2 (2x2) cols 3,4.
    
    # 2.2.1: A_R B_R2_1 (2x2 x 2x1) Rank 4
    # B_R2_1 is B[1,2], B[2,2].
    # V indices: 1*5+2=7, 2*5+2=12.
    # Result C col 2.
    # C_02 (i=0, k=2) -> W[2*2+0]=4
    # C_12 (i=1, k=2) -> W[2*2+1]=5
    
    # A_00 B_00
    add_term(1, 1, 7, 1, 4, 1)
    # A_01 B_10
    add_term(2, 1, 12, 1, 4, 1)
    # A_10 B_00
    add_term(4, 1, 7, 1, 5, 1)
    # A_11 B_10
    add_term(5, 1, 12, 1, 5, 1)
    
    # 2.2.2: A_R B_R2_2 (2x2 x 2x2) Strassen Rank 7
    # B_R2_2 is B[1,3], B[1,4], B[2,3], B[2,4].
    # V indices:
    # B_00 (orig 1,3) -> 1*5+3=8
    # B_01 (orig 1,4) -> 1*5+4=9
    # B_10 (orig 2,3) -> 2*5+3=13
    # B_11 (orig 2,4) -> 2*5+4=14
    
    # Result C cols 3,4.
    # C_03 (i=0, k=3) -> W[3*2+0]=6
    # C_04 (i=0, k=4) -> W[4*2+0]=8
    # C_13 (i=1, k=3) -> W[3*2+1]=7
    # C_14 (i=1, k=4) -> W[4*2+1]=9
    
    # M1: (A00+A11)(B00+B11) -> C03(1), C04(1), C13(1), C14(1)
    add_strassen_term(
        [(1, 1), (5, 1)], 
        [(8, 1), (14, 1)], 
        [(6, 1), (8, 1), (7, 1), (9, 1)]
    )
    
    # M2: (A10+A11)B00 -> C13(1), C14(1)
    add_strassen_term(
        [(4, 1), (5, 1)], 
        [(8, 1)], 
        [(7, 1), (9, 1)]
    )
    
    # M3: A00(B01-B11) -> C04(1), C14(1)
    add_strassen_term(
        [(1, 1)], 
        [(9, 1), (14, -1)], 
        [(8, 1), (9, 1)]
    )
    
    # M4: A11(B10-B00) -> C13(1), C14(1)
    add_strassen_term(
        [(5, 1)], 
        [(13, 1), (8, -1)], 
        [(7, 1), (9, 1)]
    )
    
    # M5: (A00+A01)B11 -> C03(1), C04(1)
    add_strassen_term(
        [(1, 1), (2, 1)], 
        [(14, 1)], 
        [(6, 1), (8, 1)]
    )
    
    # M6: (A10-A00)(B00+B01) -> C14(1)
    add_strassen_term(
        [(4, 1), (1, -1)], 
        [(8, 1), (9, 1)], 
        [(9, 1)]
    )
    
    # M7: (A01-A11)(B10+B11) -> C03(1)
    add_strassen_term(
        [(2, 1), (5, -1)], 
        [(13, 1), (14, 1)], 
        [(6, 1)]
    )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
