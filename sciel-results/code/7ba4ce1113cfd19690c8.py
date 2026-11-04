import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    
    # Define A blocks indices in U (8 entries: i*m+j)
    # A is 2x4.
    # A11: A[0,0], A[0,1] -> indices 0, 1
    # A12: A[0,2], A[0,3] -> indices 2, 3
    # A21: A[1,0], A[1,1] -> indices 4, 5
    # A22: A[1,2], A[1,3] -> indices 6, 7
    
    # Define B blocks indices in V (32 entries: j*p+k)
    # B is 4x8.
    # B11: rows 0,1; cols 0-3. Indices:
    # (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3
    # (1,0)=8, (1,1)=9, (1,2)=10, (1,3)=11
    # B12: rows 0,1; cols 4-7. Indices:
    # (0,4)=4, (0,5)=5, (0,6)=6, (0,7)=7
    # (1,4)=12, (1,5)=13, (1,6)=14, (1,7)=15
    # B21: rows 2,3; cols 0-3. Indices:
    # (2,0)=16, (2,1)=17, (2,2)=18, (2,3)=19
    # (3,0)=24, (3,1)=25, (3,2)=26, (3,3)=27
    # B22: rows 2,3; cols 4-7. Indices:
    # (2,4)=20, (2,5)=21, (2,6)=22, (2,7)=23
    # (3,4)=28, (3,5)=29, (3,6)=30, (3,7)=31
    
    # Define C blocks indices in W (16 entries: k*n+i)
    # C is 2x8.
    # C11: i=0, k=0-3. Indices: k*n+i = 0, 1, 2, 3
    # C12: i=0, k=4-7. Indices: 4, 5, 6, 7
    # C21: i=1, k=0-3. Indices: 0*2+1=1, 1*2+1=3, 2*2+1=5, 3*2+1=7 -> 1, 3, 5, 7
    # C22: i=1, k=4-7. Indices: 4*2+1=9, 5*2+1=11, 6*2+1=13, 7*2+1=15 -> 9, 11, 13, 15
    
    # Strassen terms
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # For each M, we have A_comb (2 coeffs) and B_comb (8 coeffs)
    # A_comb is vector of length 2 (for the 1x2 matrix)
    # B_comb is matrix of 2x4 (for the 2x4 matrix)
    
    # We will iterate over the 7 terms.
    # For each term, we define the A indices and coeffs, and B indices and coeffs.
    # Then we expand into 8 scalar mults.
    
    terms = []
    
    # Helper to add a term
    # a_indices: list of (u_idx, coeff)
    # b_indices: list of (v_idx, coeff)
    # c_indices: list of (w_idx, coeff)
    
    # M1
    # A: A11 (0,1), A22 (6,7) with coeff 1
    # B: B11 (0,1,2,3,8,9,10,11), B22 (20,21,22,23,28,29,30,31) with coeff 1
    # C: C11 (0,1,2,3), C22 (9,11,13,15) with coeff 1
    terms.append({
        'a': [(0, 1), (1, 1), (6, 1), (7, 1)],
        'b': [(0, 1), (1, 1), (2, 1), (3, 1), (8, 1), (9, 1), (10, 1), (11, 1),
              (20, 1), (21, 1), (22, 1), (23, 1), (28, 1), (29, 1), (30, 1), (31, 1)],
        'c': [(0, 1), (1, 1), (2, 1), (3, 1), (9, 1), (11, 1), (13, 1), (15, 1)]
    })
    
    # M2
    # A: A21 (4,5), A22 (6,7) with coeff 1
    # B: B11
    # C: C21 (1,3,5,7) with coeff 1
    terms.append({
        'a': [(4, 1), (5, 1), (6, 1), (7, 1)],
        'b': [(0, 1), (1, 1), (2, 1), (3, 1), (8, 1), (9, 1), (10, 1), (11, 1)],
        'c': [(1, 1), (3, 1), (5, 1), (7, 1)]
    })
    
    # M3
    # A: A11
    # B: B12 (4,5,6,7,12,13,14,15) coeff 1, B22 coeff -1
    # C: C12 (4,5,6,7) with coeff 1
    terms.append({
        'a': [(0, 1), (1, 1)],
        'b': [(4, 1), (5, 1), (6, 1), (7, 1), (12, 1), (13, 1), (14, 1), (15, 1),
              (20, -1), (21, -1), (22, -1), (23, -1), (28, -1), (29, -1), (30, -1), (31, -1)],
        'c': [(4, 1), (5, 1), (6, 1), (7, 1)]
    })
    
    # M4
    # A: A22
    # B: B21 (16,17,18,19,24,25,26,27) coeff 1, B11 coeff -1
    # C: C11 coeff -1, C21 coeff 1
    terms.append({
        'a': [(6, 1), (7, 1)],
        'b': [(16, 1), (17, 1), (18, 1), (19, 1), (24, 1), (25, 1), (26, 1), (27, 1),
              (0, -1), (1, -1), (2, -1), (3, -1), (8, -1), (9, -1), (10, -1), (11, -1)],
        'c': [(0, -1), (1, -1), (2, -1), (3, -1), (1, 1), (3, 1), (5, 1), (7, 1)]
    })
    
    # M5
    # A: A11, A12
    # B: B22
    # C: C11 coeff -1, C12 coeff 1
    terms.append({
        'a': [(0, 1), (1, 1), (2, 1), (3, 1)],
        'b': [(20, 1), (21, 1), (22, 1), (23, 1), (28, 1), (29, 1), (30, 1), (31, 1)],
        'c': [(0, -1), (1, -1), (2, -1), (3, -1), (4, 1), (5, 1), (6, 1), (7, 1)]
    })
    
    # M6
    # A: A21 coeff 1, A11 coeff -1
    # B: B11 coeff 1, B12 coeff 1
    # C: C22 coeff 1
    terms.append({
        'a': [(4, 1), (5, 1), (0, -1), (1, -1)],
        'b': [(0, 1), (1, 1), (2, 1), (3, 1), (8, 1), (9, 1), (10, 1), (11, 1),
              (4, 1), (5, 1), (6, 1), (7, 1), (12, 1), (13, 1), (14, 1), (15, 1)],
        'c': [(9, 1), (11, 1), (13, 1), (15, 1)]
    })
    
    # M7
    # A: A12 coeff 1, A22 coeff -1
    # B: B21 coeff 1, B22 coeff 1
    # C: C11 coeff 1, C22 coeff -1
    terms.append({
        'a': [(2, 1), (3, 1), (6, -1), (7, -1)],
        'b': [(16, 1), (17, 1), (18, 1), (19, 1), (24, 1), (25, 1), (26, 1), (27, 1),
              (20, 1), (21, 1), (22, 1), (23, 1), (28, 1), (29, 1), (30, 1), (31, 1)],
        'c': [(0, 1), (1, 1), (2, 1), (3, 1), (9, -1), (11, -1), (13, -1), (15, -1)]
    })
    
    # Now expand each term into 8 scalar multiplications
    # For a term, A is 1x2, B is 2x4.
    # The 8 mults correspond to (j,k) in 0..1, 0..3.
    # U index for A_j: 
    #   If A_j is a linear combo of A_entries, we add to U.
    #   The "local" A vector is [a0, a1].
    #   The "local" B matrix is [[b00, b01, b02, b03], [b10, b11, b12, b13]].
    #   The "local" C vector is [c0, c1, c2, c3].
    #   c_k = a0*b0k + a1*b1k.
    
    # We iterate j in 0,1 and k in 0..3.
    # For each (j,k), we have:
    # u_vec = e_j (1 at j, 0 else)
    # v_mat = E_jk (1 at j,k, 0 else)
    # w_vec = e_k (1 at k, 0 else)
    
    # We add:
    # U[global_A_idx] += coeff_A * u_vec[j]
    # V[global_B_idx] += coeff_B * v_mat[j,k]
    # W[global_C_idx] += coeff_C * w_vec[k]
    
    for term in terms:
        a_coeffs = term['a']
        b_coeffs = term['b']
        c_coeffs = term['c']
        
        # Map local indices to global
        # Local A has 2 entries.
        # We need to know which global A index corresponds to local 0 and local 1.
        # Actually, the term definition already gives global indices and coeffs.
        # We just need to distribute the "1"s from the standard basis vectors.
        
        # For j in 0,1:
        for j in range(2):
            for k in range(4):
                # u_vec has 1 at j
                # v_mat has 1 at (j,k)
                # w_vec has 1 at k
                
                # Add to U
                for a_idx, a_coeff in a_coeffs:
                    # Only if this global A index is part of local A[j]?
                    # No, the term 'a' contains ALL global A indices involved in the linear combo for local A.
                    # Wait. The local A is [a0, a1].
                    # a0 is a linear combo of global A entries.
                    # a1 is a linear combo of global A entries.
                    # My 'a' list contains (global_idx, coeff) for the ENTIRE local A vector?
                    # No, I listed them flat. I need to know which ones are a0 and which are a1.
                    
                    # Let's restructure the term definition to be explicit.
                    pass

    # Redefine terms with explicit local structure
    # local_a: list of 2 items, each is list of (global_idx, coeff)
    # local_b: list of 2 rows, each row is list of 4 items (global_idx, coeff)
    # local_c: list of 4 items, each is list of (global_idx, coeff)
    
    def make_term(a0, a1, b00, b01, b02, b03, b10, b11, b12, b13, c0, c1, c2, c3):
        return {
            'a0': a0, 'a1': a1,
            'b00': b00, 'b01': b01, 'b02': b02, 'b03': b03,
            'b10': b10, 'b11': b11, 'b12': b12, 'b13': b13,
            'c0': c0, 'c1': c1, 'c2': c2, 'c3': c3
        }
    
    # A11: [(0,1), (1,1)]
    # A12: [(2,1), (3,1)]
    # A21: [(4,1), (5,1)]
    # A22: [(6,1), (7,1)]
    
    # B11: [[(0,1),(1,1),(2,1),(3,1)], [(8,1),(9,1),(10,1),(11,1)]]
    # B12: [[(4,1),(5,1),(6,1),(7,1)], [(12,1),(13,1),(14,1),(15,1)]]
    # B21: [[(16,1),(17,1),(18,1),(19,1)], [(24,1),(25,1),(26,1),(27,1)]]
    # B22: [[(20,1),(21,1),(22,1),(23,1)], [(28,1),(29,1),(30,1),(31,1)]]
    
    # C11: [(0,1),(1,1),(2,1),(3,1)]
    # C12: [(4,1),(5,1),(6,1),(7,1)]
    # C21: [(1,1),(3,1),(5,1),(7,1)]
    # C22: [(9,1),(11,1),(13,1),(15,1)]
    
    A11 = [(0,1), (1,1)]
    A12 = [(2,1), (3,1)]
    A21 = [(4,1), (5,1)]
    A22 = [(6,1), (7,1)]
    
    B11 = [[(0,1),(1,1),(2,1),(3,1)], [(8,1),(9,1),(10,1),(11,1)]]
    B12 = [[(4,1),(5,1),(6,1),(7,1)], [(12,1),(13,1),(14,1),(15,1)]]
    B21 = [[(16,1),(17,1),(18,1),(19,1)], [(24,1),(25,1),(26,1),(27,1)]]
    B22 = [[(20,1),(21,1),(22,1),(23,1)], [(28,1),(29,1),(30,1),(31,1)]]
    
    C11 = [(0,1),(1,1),(2,1),(3,1)]
    C12 = [(4,1),(5,1),(6,1),(7,1)]
    C21 = [(1,1),(3,1),(5,1),(7,1)]
    C22 = [(9,1),(11,1),(13,1),(15,1)]
    
    def add_scalar(a_list, b_list, c_list, sign=1):
        nonlocal col
        # a_list: list of (idx, coeff) for local A[j]
        # b_list: list of (idx, coeff) for local B[j,k]
        # c_list: list of (idx, coeff) for local C[k]
        
        for a_idx, a_coeff in a_list:
            U[a_idx][col] += sign * a_coeff
        for b_idx, b_coeff in b_list:
            V[b_idx][col] += sign * b_coeff
        for c_idx, c_coeff in c_list:
            W[c_idx][col] += sign * c_coeff
        col += 1

    # M1: (A11+A22)(B11+B22)
    # a0 = A11[0] + A22[0] = [(0,1),(6,1)]
    # a1 = A11[1] + A22[1] = [(1,1),(7,1)]
    # b00 = B11[0][0] + B22[0][0] = [(0,1),(20,1)]
    # ...
    # c0 = C11[0] + C22[0] = [(0,1),(9,1)]
    # ...
    
    # We will write a helper to combine lists
    def comb(list1, list2, s1=1, s2=1):
        res = []
        for idx, c in list1: res.append((idx, c*s1))
        for idx, c in list2: res.append((idx, c*s2))
        return res
    
    # M1
    a0 = comb(A11[0], A22[0])
    a1 = comb(A11[1], A22[1])
    b00 = comb(B11[0][0], B22[0][0])
    b01 = comb(B11[0][1], B22[0][1])
    b02 = comb(B11[0][2], B22[0][2])
    b03 = comb(B11[0][3], B22[0][3])
    b10 = comb(B11[1][0], B22[1][0])
    b11 = comb(B11[1][1], B22[1][1])
    b12 = comb(B11[1][2], B22[1][2])
    b13 = comb(B11[1][3], B22[1][3])
    c0 = comb(C11[0], C22[0])
    c1 = comb(C11[1], C22[1])
    c2 = comb(C11[2], C22[2])
    c3 = comb(C11[3], C22[3])
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    # M2: (A21+A22)B11
    a0 = comb(A21[0], A22[0])
    a1 = comb(A21[1], A22[1])
    b00 = B11[0][0]
    b01 = B11[0][1]
    b02 = B11[0][2]
    b03 = B11[0][3]
    b10 = B11[1][0]
    b11 = B11[1][1]
    b12 = B11[1][2]
    b13 = B11[1][3]
    c0 = C21[0]
    c1 = C21[1]
    c2 = C21[2]
    c3 = C21[3]
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    # M3: A11(B12-B22)
    a0 = A11[0]
    a1 = A11[1]
    b00 = comb(B12[0][0], B22[0][0], 1, -1)
    b01 = comb(B12[0][1], B22[0][1], 1, -1)
    b02 = comb(B12[0][2], B22[0][2], 1, -1)
    b03 = comb(B12[0][3], B22[0][3], 1, -1)
    b10 = comb(B12[1][0], B22[1][0], 1, -1)
    b11 = comb(B12[1][1], B22[1][1], 1, -1)
    b12 = comb(B12[1][2], B22[1][2], 1, -1)
    b13 = comb(B12[1][3], B22[1][3], 1, -1)
    c0 = C12[0]
    c1 = C12[1]
    c2 = C12[2]
    c3 = C12[3]
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    # M4: A22(B21-B11)
    a0 = A22[0]
    a1 = A22[1]
    b00 = comb(B21[0][0], B11[0][0], 1, -1)
    b01 = comb(B21[0][1], B11[0][1], 1, -1)
    b02 = comb(B21[0][2], B11[0][2], 1, -1)
    b03 = comb(B21[0][3], B11[0][3], 1, -1)
    b10 = comb(B21[1][0], B11[1][0], 1, -1)
    b11 = comb(B21[1][1], B11[1][1], 1, -1)
    b12 = comb(B21[1][2], B11[1][2], 1, -1)
    b13 = comb(B21[1][3], B11[1][3], 1, -1)
    c0 = comb(C11[0], C21[0], -1, 1)
    c1 = comb(C11[1], C21[1], -1, 1)
    c2 = comb(C11[2], C21[2], -1, 1)
    c3 = comb(C11[3], C21[3], -1, 1)
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    # M5: (A11+A12)B22
    a0 = comb(A11[0], A12[0])
    a1 = comb(A11[1], A12[1])
    b00 = B22[0][0]
    b01 = B22[0][1]
    b02 = B22[0][2]
    b03 = B22[0][3]
    b10 = B22[1][0]
    b11 = B22[1][1]
    b12 = B22[1][2]
    b13 = B22[1][3]
    c0 = comb(C11[0], C12[0], -1, 1)
    c1 = comb(C11[1], C12[1], -1, 1)
    c2 = comb(C11[2], C12[2], -1, 1)
    c3 = comb(C11[3], C12[3], -1, 1)
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    # M6: (A21-A11)(B11+B12)
    a0 = comb(A21[0], A11[0], 1, -1)
    a1 = comb(A21[1], A11[1], 1, -1)
    b00 = comb(B11[0][0], B12[0][0])
    b01 = comb(B11[0][1], B12[0][1])
    b02 = comb(B11[0][2], B12[0][2])
    b03 = comb(B11[0][3], B12[0][3])
    b10 = comb(B11[1][0], B12[1][0])
    b11 = comb(B11[1][1], B12[1][1])
    b12 = comb(B11[1][2], B12[1][2])
    b13 = comb(B11[1][3], B12[1][3])
    c0 = C22[0]
    c1 = C22[1]
    c2 = C22[2]
    c3 = C22[3]
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    # M7: (A12-A22)(B21+B22)
    a0 = comb(A12[0], A22[0], 1, -1)
    a1 = comb(A12[1], A22[1], 1, -1)
    b00 = comb(B21[0][0], B22[0][0])
    b01 = comb(B21[0][1], B22[0][1])
    b02 = comb(B21[0][2], B22[0][2])
    b03 = comb(B21[0][3], B22[0][3])
    b10 = comb(B21[1][0], B22[1][0])
    b11 = comb(B21[1][1], B22[1][1])
    b12 = comb(B21[1][2], B22[1][2])
    b13 = comb(B21[1][3], B22[1][3])
    c0 = comb(C11[0], C22[0], 1, -1)
    c1 = comb(C11[1], C22[1], 1, -1)
    c2 = comb(C11[2], C22[2], 1, -1)
    c3 = comb(C11[3], C22[3], 1, -1)
    
    add_scalar(a0, b00, c0)
    add_scalar(a0, b01, c1)
    add_scalar(a0, b02, c2)
    add_scalar(a0, b03, c3)
    add_scalar(a1, b10, c0)
    add_scalar(a1, b11, c1)
    add_scalar(a1, b12, c2)
    add_scalar(a1, b13, c3)
    
    return {"U": U, "V": V, "W": W}

