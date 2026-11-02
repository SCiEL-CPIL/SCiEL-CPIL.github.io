"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_map, u_vals, v_map, v_vals, w_map, w_vals):
        nonlocal col
        for i, val in zip(u_map, u_vals):
            if val != 0:
                U[i][col] += val
        for i, val in zip(v_map, v_vals):
            if val != 0:
                V[i][col] += val
        for i, val in zip(w_map, w_vals):
            if val != 0:
                W[i][col] += val
        col += 1

    # Strassen 2x2x2 terms (u, v, w)
    # u: [a,b,c,d]
    # v: [e,f,g,h]
    # w: [C00, C01, C10, C11]
    strassen_terms = [
        ([1,0,0,1], [1,1,1,1], [1,0,0,-1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),
    ]
    
    # Schoolbook 2x2x1 terms (u, v, w)
    # u: [a,b,c,d]
    # v: [e,g]
    # w: [C0, C1]
    schoolbook_terms = [
        ([1,0,0,0], [1,0], [1,0]),
        ([0,1,0,0], [0,1], [1,0]),
        ([0,0,1,0], [1,0], [0,1]),
        ([0,0,0,1], [0,1], [0,1]),
    ]

    # Maps for A_L (cols 0,1 of A)
    # A is 2x4. Indices: 0,1,2,3 (row 0), 4,5,6,7 (row 1)
    # A_L: A00, A01, A10, A11 -> 0, 1, 4, 5
    map_A_L = [0, 1, 4, 5]
    
    # Maps for A_R (cols 2,3 of A)
    # A_R: A02, A03, A12, A13 -> 2, 3, 6, 7
    map_A_R = [2, 3, 6, 7]
    
    # Process A_L * B_T
    # B_T is rows 0,1 of B (4x5).
    # B_T cols 0,1: B00, B01, B10, B11 -> 0, 1, 5, 6
    map_B_T_01 = [0, 1, 5, 6]
    # B_T cols 2,3: B02, B03, B12, B13 -> 2, 3, 7, 8
    map_B_T_23 = [2, 3, 7, 8]
    # B_T col 4: B04, B14 -> 4, 9
    map_B_T_4 = [4, 9]
    
    # C is 2x5. Indices: 0..4 (row 0), 5..9 (row 1)
    # C cols 0,1: C00, C01, C10, C11 -> 0, 1, 5, 6
    map_C_01 = [0, 1, 5, 6]
    # C cols 2,3: C02, C03, C12, C13 -> 2, 3, 7, 8
    map_C_23 = [2, 3, 7, 8]
    # C col 4: C04, C14 -> 4, 9
    map_C_4 = [4, 9]
    
    # A_L * B_T (cols 0,1) using Strassen
    for u, v, w in strassen_terms:
        add_term(map_A_L, u, map_B_T_01, v, map_C_01, w)
        
    # A_L * B_T (cols 2,3) using Strassen
    for u, v, w in strassen_terms:
        add_term(map_A_L, u, map_B_T_23, v, map_C_23, w)
        
    # A_L * B_T (col 4) using Schoolbook
    for u, v, w in schoolbook_terms:
        add_term(map_A_L, u, map_B_T_4, v, map_C_4, w)
        
    # Process A_R * B_B
    # B_B is rows 2,3 of B.
    # B_B cols 0,1: B20, B21, B30, B31 -> 10, 11, 15, 16
    map_B_B_01 = [10, 11, 15, 16]
    # B_B cols 2,3: B22, B23, B32, B33 -> 12, 13, 17, 18
    map_B_B_23 = [12, 13, 17, 18]
    # B_B col 4: B24, B34 -> 14, 19
    map_B_B_4 = [14, 19]
    
    # A_R * B_B (cols 0,1) using Strassen
    for u, v, w in strassen_terms:
        add_term(map_A_R, u, map_B_B_01, v, map_C_01, w)
        
    # A_R * B_B (cols 2,3) using Strassen
    for u, v, w in strassen_terms:
        add_term(map_A_R, u, map_B_B_23, v, map_C_23, w)
        
    # A_R * B_B (col 4) using Schoolbook
    for u, v, w in schoolbook_terms:
        add_term(map_A_R, u, map_B_B_4, v, map_C_4, w)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
