"""Rank-28 decomposition of the 2x4 * 4x4 matrix multiplication tensor."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # We decompose M_{2,4,4} into 4 independent M_{2,2,2} multiplications,
    # each done via Strassen (rank 7). Total rank = 4 * 7 = 28.
    #
    # A is 2x4: A = [A0 | A1] where A0, A1 are 2x2.
    # B is 4x4: B = [[B00, B01], [B10, B11]] where each is 2x2.
    # C = A*B is 2x4: C = [C0 | C1] where C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11.
    #
    # So we need 4 independent 2x2x2x2 multiplications:
    #   M1 = A0 * B00 -> contributes to C0
    #   M2 = A1 * B10 -> contributes to C0
    #   M3 = A0 * B01 -> contributes to C1
    #   M4 = A1 * B11 -> contributes to C1
    #
    # For each 2x2x2x2 multiplication, Strassen gives 7 rank-1 terms.
    #
    # Strassen formulas for X*Y = Z (all 2x2):
    #   P1 = (X00+X11)(Y00+Y11)
    #   P2 = (X10+X11)(Y00)
    #   P3 = (X00)(Y01-Y11)
    #   P4 = (X11)(Y10-Y00)
    #   P5 = (X00+X01)(Y11)
    #   P6 = (X10-X00)(Y00+Y01)
    #   P7 = (X01-X11)(Y10+Y11)
    #
    # Result:
    #   Z00 = P1 + P4 - P5 + P7
    #   Z01 = P3 + P5
    #   Z10 = P2 + P4
    #   Z11 = P1 - P2 + P3 + P6
    #
    # In tensor decomposition terms, for each product Pi = u_i^T (x ⊗ y) v_i:
    #   U row picks elements from x (the A-part), V row picks from y (the B-part),
    #   W row places into z (the C-part).
    #
    # Dimensions: U is (n*m, rank) = (8, 28), V is (m*p, rank) = (16, 28), W is (p*n, rank) = (8, 28)
    # U index: i*m + j where i is row of A (0,1), j is col of A (0,1,2,3)
    # V index: j*p + k where j is row of B (0,1,2,3), k is col of B (0,1,2,3)
    # W index: k*n + i where k is col of C (0,1,2,3), i is row of C (0,1)
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    # For each of the 4 sub-multiplications, we have 7 Strassen products.
    # Sub-multiplication s (s=0,1,2,3):
    #   s=0: A0*B00 -> C0 (cols 0,1 of C)
    #     A part: rows 0,1 of A, cols 0,1 -> A indices: (0,0),(0,1),(1,0),(1,1)
    #     B part: rows 0,1 of B, cols 0,1 -> B indices: (0,0),(0,1),(1,0),(1,1)
    #     C part: rows 0,1 of C, cols 0,1 -> C indices: (0,0),(0,1),(1,0),(1,1)
    #   s=1: A1*B10 -> C0 (cols 0,1 of C)
    #     A part: rows 0,1 of A, cols 2,3 -> A indices: (0,2),(0,3),(1,2),(1,3)
    #     B part: rows 2,3 of B, cols 0,1 -> B indices: (2,0),(2,1),(3,0),(3,1)
    #     C part: rows 0,1 of C, cols 0,1 -> C indices: (0,0),(0,1),(1,0),(1,1)
    #   s=2: A0*B01 -> C1 (cols 2,3 of C)
    #     A part: rows 0,1 of A, cols 0,1 -> A indices: (0,0),(0,1),(1,0),(1,1)
    #     B part: rows 0,1 of B, cols 2,3 -> B indices: (0,2),(0,3),(1,2),(1,3)
    #     C part: rows 0,1 of C, cols 2,3 -> C indices: (0,2),(0,3),(1,2),(1,3)
    #   s=3: A1*B11 -> C1 (cols 2,3 of C)
    #     A part: rows 0,1 of A, cols 2,3 -> A indices: (0,2),(0,3),(1,2),(1,3)
    #     B part: rows 2,3 of B, cols 2,3 -> B indices: (2,2),(2,3),(3,2),(3,3)
    #     C part: rows 0,1 of C, cols 2,3 -> C indices: (0,2),(0,3),(1,2),(1,3)
    
    # Define the Strassen coefficients for the 7 products.
    # Each product t (0..6) has:
    #   u_coefs: coefficients for the 4 elements of X (X00, X01, X10, X11)
    #   v_coefs: coefficients for the 4 elements of Y (Y00, Y01, Y10, Y11)
    #   w_coefs: coefficients for the 4 elements of Z (Z00, Z01, Z10, Z11)
    
    strassen = [
        # t=0: P1 = (X00+X11)(Y00+Y11), Z: Z00+=1, Z11+=1
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # t=1: P2 = (X10+X11)(Y00), Z: Z10+=1, Z11-=1
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        # t=2: P3 = (X00)(Y01-Y11), Z: Z01+=1, Z11+=1
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        # t=3: P4 = (X11)(Y10-Y00), Z: Z00+=1, Z10+=1
        {'u': [0, 0, 0, 1], 'v': [0, 0, 1, -1], 'w': [1, 0, 1, 0]},
        # t=4: P5 = (X00+X01)(Y11), Z: Z00-=1, Z01+=1
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        # t=5: P6 = (X10-X00)(Y00+Y01), Z: Z11+=1
        {'u': [0, 0, 1, -1], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # t=6: P7 = (X01-X11)(Y10+Y11), Z: Z00+=1
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Define the 4 sub-multiplications with their index mappings.
    # For each sub-mult, we map local indices (0,1,2,3) to global indices.
    # Local X indices: X00->(ri,ci), X01->(ri,ci+1), X10->(ri+1,ci), X11->(ri+1,ci+1)
    # Local Y indices: Y00->(rj,cj), Y01->(rj,cj+1), Y10->(rj+1,cj), Y11->(rj+1,cj+1)
    # Local Z indices: Z00->(ri,czi), Z01->(ri,czi+1), Z10->(ri+1,czi), Z11->(ri+1,czi+1)
    
    submults = [
        # s=0: A0*B00 -> C0
        {'a_row': 0, 'a_col': 0, 'b_row': 0, 'b_col': 0, 'c_col': 0},
        # s=1: A1*B10 -> C0
        {'a_row': 0, 'a_col': 2, 'b_row': 2, 'b_col': 0, 'c_col': 0},
        # s=2: A0*B01 -> C1
        {'a_row': 0, 'a_col': 0, 'b_row': 0, 'b_col': 2, 'c_col': 2},
        # s=3: A1*B11 -> C1
        {'a_row': 0, 'a_col': 2, 'b_row': 2, 'b_col': 2, 'c_col': 2},
    ]
    
    for s_idx, sm in enumerate(submults):
        ar = sm['a_row']  # starting row in A (always 0 since n=2)
        ac = sm['a_col']  # starting col in A
        br = sm['b_row']  # starting row in B
        bc = sm['b_col']  # starting col in B
        cc = sm['c_col']  # starting col in C
        
        for t_idx, st in enumerate(strassen):
            col = s_idx * 7 + t_idx  # 0..27
            
            # Fill U: for each of the 4 local X elements, add coefficient to the corresponding A entry
            # Local X element (li, lj) where li in {0,1}, lj in {0,1}
            # Global A index: (ar+li) * m + (ac + lj)
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj  # X00=0, X01=1, X10=2, X11=3
                    coef = st['u'][local_idx]
                    if coef != 0:
                        a_global = (ar + li) * m + (ac + lj)
                        U[a_global][col] += coef
            
            # Fill V: for each of the 4 local Y elements
            # Global B index: (br+li) * p + (bc + lj)
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    coef = st['v'][local_idx]
                    if coef != 0:
                        b_global = (br + li) * p + (bc + lj)
                        V[b_global][col] += coef
            
            # Fill W: for each of the 4 local Z elements
            # Global C index: (cc + lj) * n + (ar + li)  # W[k*n+i] where k=col, i=row
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    coef = st['w'][local_idx]
                    if coef != 0:
                        c_global = (cc + lj) * n + (ar + li)
                        W[c_global][col] += coef
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
