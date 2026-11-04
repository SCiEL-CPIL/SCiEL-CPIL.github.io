import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_term(U_indices, V_indices, W_indices, coeffs=1):
        nonlocal col_idx
        if col_idx >= rank:
            raise Exception("Rank exceeded")
        for i in U_indices:
            U[i][col_idx] += coeffs
        for j in V_indices:
            V[j][col_idx] += coeffs
        for k in W_indices:
            W[k][col_idx] += coeffs
        col_idx += 1

    def add_schoolbook(A_block, B_block, C_block):
        # A_block: list of (r, c) in A
        # B_block: list of (r, c) in B
        # C_block: list of (r, c) in C
        # Assumes A is 1x1, B is 1x1, C is 1x1? No, general.
        # This function adds schoolbook terms for a set of pairs.
        pass

    # Helper to get flat index
    def a_idx(i, j): return i * m + j
    def b_idx(i, j): return i * p + j
    def c_idx(i, j): return j * n + i # W[k*n+i, r] -> index in W is k*n+i. 
    # Wait, W shape is (p*n, rank).
    # Output factor follows W[k*n+i, r].
    # So W index is k*n + i.
    # k is col of B (0..p-1), i is row of A (0..n-1).
    
    # Strassen 2x2x2
    # A: [[a, b], [c, d]] -> indices a_idx(i1, j1), a_idx(i1, j2), a_idx(i2, j1), a_idx(i2, j2)
    # B: [[e, f], [g, h]] -> indices b_idx(k1, l1), b_idx(k1, l2), b_idx(k2, l1), b_idx(k2, l2)
    # C: [[c11, c12], [c21, c22]] -> indices c_idx(i1, l1), c_idx(i1, l2), c_idx(i2, l1), c_idx(i2, l2)
    
    def add_strassen(i1, i2, j1, j2, k1, k2, l1, l2):
        # A indices
        a = a_idx(i1, j1)
        b = a_idx(i1, j2)
        c = a_idx(i2, j1)
        d = a_idx(i2, j2)
        
        # B indices
        e = b_idx(k1, l1)
        f = b_idx(k1, l2)
        g = b_idx(k2, l1)
        h = b_idx(k2, l2)
        
        # C indices
        c11 = c_idx(i1, l1)
        c12 = c_idx(i1, l2)
        c21 = c_idx(i2, l1)
        c22 = c_idx(i2, l2)
        
        # M1 = (a+d)(e+h) -> C11, C22
        add_term([a, d], [e, h], [c11, c22], 1)
        
        # M2 = (c+d)e -> C21, C22
        add_term([c, d], [e], [c21, c22], 1)
        
        # M3 = a(f-h) -> C12, C22
        add_term([a], [f, h], [c12, c22], 1)
        # Note: f-h means coeff 1 for f, -1 for h.
        # My add_term takes single coeff. Need to handle signs.
        
    # Refined add_term
    def add_term_signed(U_idx_coeff, V_idx_coeff, W_idx_coeff):
        nonlocal col_idx
        if col_idx >= rank:
            return
        for idx, coeff in U_idx_coeff:
            U[idx][col_idx] += coeff
        for idx, coeff in V_idx_coeff:
            V[idx][col_idx] += coeff
        for idx, coeff in W_idx_coeff:
            W[idx][col_idx] += coeff
        col_idx += 1

    def add_strassen(i1, i2, j1, j2, k1, k2, l1, l2):
        a = a_idx(i1, j1)
        b = a_idx(i1, j2)
        c = a_idx(i2, j1)
        d = a_idx(i2, j2)
        
        e = b_idx(k1, l1)
        f = b_idx(k1, l2)
        g = b_idx(k2, l1)
        h = b_idx(k2, l2)
        
        c11 = c_idx(i1, l1)
        c12 = c_idx(i1, l2)
        c21 = c_idx(i2, l1)
        c22 = c_idx(i2, l2)
        
        # M1 = (a+d)(e+h)
        add_term_signed([(a,1),(d,1)], [(e,1),(h,1)], [(c11,1),(c22,1)])
        
        # M2 = (c+d)e
        add_term_signed([(c,1),(d,1)], [(e,1)], [(c21,1),(c22,1)])
        
        # M3 = a(f-h)
        add_term_signed([(a,1)], [(f,1),(h,-1)], [(c12,1),(c22,1)])
        
        # M4 = d(g-e)
        add_term_signed([(d,1)], [(g,1),(e,-1)], [(c11,1),(c22,1)])
        
        # M5 = (a+b)h
        add_term_signed([(a,1),(b,1)], [(h,1)], [(c11,-1),(c21,1)])
        # C11 -= M5, C21 += M5
        
        # M6 = (c-a)(e+f)
        add_term_signed([(c,1),(a,-1)], [(e,1),(f,1)], [(c22,1)])
        # C22 += M6
        
        # M7 = (b-d)(g+h)
        add_term_signed([(b,1),(d,-1)], [(g,1),(h,1)], [(c11,1),(c12,1)])
        # C11 += M7, C12 += M7

    def add_schoolbook_2x2x1(i1, i2, j, k1, k2, l):
        # A is 2x1, B is 2x1, C is 2x1
        # C1 = A1 B1 + A2 B2
        a1 = a_idx(i1, j)
        a2 = a_idx(i2, j)
        b1 = b_idx(k1, l)
        b2 = b_idx(k2, l)
        c1 = c_idx(i1, l)
        c2 = c_idx(i2, l)
        
        add_term_signed([(a1,1)], [(b1,1)], [(c1,1)])
        add_term_signed([(a2,1)], [(b2,1)], [(c2,1)])

    def add_schoolbook_1x2x2(i, j1, j2, k1, k2, l1, l2):
        # A is 1x2, B is 2x2, C is 1x2
        a1 = a_idx(i, j1)
        a2 = a_idx(i, j2)
        
        e = b_idx(k1, l1)
        f = b_idx(k1, l2)
        g = b_idx(k2, l1)
        h = b_idx(k2, l2)
        
        c1 = c_idx(i, l1)
        c2 = c_idx(i, l2)
        
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        add_term_signed([(a1,1)], [(e,1)], [(c1,1)])
        add_term_signed([(a2,1)], [(g,1)], [(c1,1)])
        add_term_signed([(a1,1)], [(f,1)], [(c2,1)])
        add_term_signed([(a2,1)], [(h,1)], [(c2,1)])

    def add_schoolbook_1x2x1(i, j, k1, k2, l):
        a = a_idx(i, j)
        b1 = b_idx(k1, l)
        b2 = b_idx(k2, l)
        c = c_idx(i, l)
        
        add_term_signed([(a,1)], [(b1,1)], [(c,1)])
        add_term_signed([(a,1)], [(b2,1)], [(c,1)])

    def add_schoolbook_2x1x2(i1, i2, j, k, l1, l2):
        a1 = a_idx(i1, j)
        a2 = a_idx(i2, j)
        b = b_idx(k, l1) # Wait, B is 2x2? No, B is 1x2?
        # A 2x1, B 1x2, C 2x2
        # C11 = A1 B1, C12 = A1 B2, C21 = A2 B1, C22 = A2 B2
        b1 = b_idx(k, l1)
        b2 = b_idx(k, l2)
        
        c11 = c_idx(i1, l1)
        c12 = c_idx(i1, l2)
        c21 = c_idx(i2, l1)
        c22 = c_idx(i2, l2)
        
        add_term_signed([(a1,1)], [(b1,1)], [(c11,1)])
        add_term_signed([(a1,1)], [(b2,1)], [(c12,1)])
        add_term_signed([(a2,1)], [(b1,1)], [(c21,1)])
        add_term_signed([(a2,1)], [(b2,1)], [(c22,1)])

    def add_schoolbook_1x1x1(i, j, k, l):
        a = a_idx(i, j)
        b = b_idx(k, l)
        c = c_idx(i, l)
        add_term_signed([(a,1)], [(b,1)], [(c,1)])

    # Main logic
    # Split A (3x5) into A1 (3x2), A2 (3x2), A3 (3x1)
    # Split B (5x7) into B1 (2x7), B2 (2x7), B3 (1x7)
    
    # Process A1 B1
    # A1 rows 0,1,2. Cols 0,1.
    # B1 rows 0,1. Cols 0..6.
    # Split B1 cols into 0,1 (2), 2,3 (2), 4,5,6 (3)
    
    # Block 1: A1 (3x2) * B1_part1 (2x2)
    # A1 rows 0,1 vs B1 cols 0,1 -> Strassen
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1)
    # A1 row 2 vs B1 cols 0,1 -> Schoolbook 1x2x2
    add_schoolbook_1x2x2(2, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A1 (3x2) * B1_part2 (2x2)
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3)
    add_schoolbook_1x2x2(2, 0, 1, 0, 1, 2, 3)
    
    # Block 3: A1 (3x2) * B1_part3 (2x3)
    # Split B1_part3 cols 4,5,6 into 4 (1) and 5,6 (2)
    # A1 rows 0,1 vs B1 col 4 -> Schoolbook 2x2x1
    add_schoolbook_2x2x1(0, 1, 0, 0, 4)
    add_schoolbook_2x2x1(0, 1, 1, 0, 4)
    # A1 rows 0,1 vs B1 cols 5,6 -> Strassen
    add_strassen(0, 1, 0, 1, 0, 1, 5, 6)
    
    # A1 row 2 vs B1 col 4 -> Schoolbook 1x2x1
    add_schoolbook_1x2x1(2, 0, 0, 4)
    add_schoolbook_1x2x1(2, 1, 0, 4)
    # A1 row 2 vs B1 cols 5,6 -> Schoolbook 1x2x2
    add_schoolbook_1x2x2(2, 0, 1, 0, 1, 5, 6)
    
    # Process A2 B2
    # A2 rows 0,1,2. Cols 2,3.
    # B2 rows 2,3. Cols 0..6.
    
    # Block 4: A2 * B2_part1 (cols 0,1)
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1)
    add_schoolbook_1x2x2(2, 2, 3, 2, 3, 0, 1)
    
    # Block 5: A2 * B2_part2 (cols 2,3)
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3)
    add_schoolbook_1x2x2(2, 2, 3, 2, 3, 2, 3)
    
    # Block 6: A2 * B2_part3 (cols 4,5,6)
    add_schoolbook_2x2x1(0, 1, 2, 2, 4)
    add_schoolbook_2x2x1(0, 1, 3, 2, 4)
    add_strassen(0, 1, 2, 3, 2, 3, 5, 6)
    
    add_schoolbook_1x2x1(2, 2, 2, 4)
    add_schoolbook_1x2x1(2, 3, 2, 4)
    add_schoolbook_1x2x2(2, 2, 3, 2, 3, 5, 6)
    
    # Process A3 B3
    # A3 rows 0,1,2. Col 4.
    # B3 row 4. Cols 0..6.
    # Schoolbook 3x1x7
    for i in range(3):
        for l in range(7):
            add_schoolbook_1x1x1(i, 4, 4, l)
            
    return {"U": U, "V": V, "W": W}
