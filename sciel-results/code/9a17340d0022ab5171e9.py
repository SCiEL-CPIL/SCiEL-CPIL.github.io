"""Improved program for 2x3x5 matrix multiplication tensor decomposition."""

def construct():
    n, m, p = 2, 3, 5
    
    # We decompose the 2x3x5 problem into:
    # 1) 2x3x3 (schoolbook, rank 18) for the first 3 columns of B
    # 2) 2x3x2 (split 3 into 1+2: rank 4 + rank 7 = 11) for the last 2 columns of B
    # Total rank = 18 + 11 = 29
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_triplet(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a single rank-1 term (triplet) to the decomposition."""
        U_list.append([0.0] * (n * m) if not U_list else U_list[-1][:])
        V_list.append([0.0] * (m * p) if not V_list else V_list[-1][:])
        W_list.append([0.0] * (p * n) if not W_list else W_list[-1][:])
        # Actually, let's build U, V, W as lists of columns
        # Each column is a vector
        pass
    
    # Let's build U, V, W as lists of column vectors
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        """Add one rank-1 tensor term."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Part 1: 2x3x3 schoolbook for columns 0,1,2 of B
    # A is 2x3, B_part1 is 3x3 (columns 0,1,2), C_part1 is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k] for k in {0,1,2}
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # But here k only goes 0,1,2 for this part
    
    for i in range(n):  # i = 0, 1
        for j in range(m):  # j = 0, 1, 2
            for k in range(3):  # k = 0, 1, 2
                u_vec = [0.0] * (n * m)
                v_vec = [0.0] * (m * p)
                w_vec = [0.0] * (p * n)
                u_vec[i * m + j] = 1.0
                v_vec[j * p + k] = 1.0  # k is 0,1,2 which are the first 3 columns
                w_vec[k * n + i] = 1.0
                add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 2x3x2 for columns 3,4 of B
    # Split B_part2 (3x2) into B_a (3x1, column 3) and B_b (3x1, column 4)
    # Actually, let's use the 1+2 split of the 3 rows of A... 
    # Wait, the directive says split 3 into 1+2. Let me think about this more carefully.
    # 
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # We can split the middle dimension (3) into 1+2.
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x2, B2 is 2x2
    # C = A1*B1 + A2*B2
    # A1*B1: 2x1x2, rank = 2*1*2 = 4 (schoolbook)
    # A2*B2: 2x2x2, rank = 7 (Strassen)
    # Total = 4 + 7 = 11
    
    # A1*B1: 2x1x2 schoolbook
    # A1 is 2x1, B1 is 1x2, C1 is 2x2
    # C1[i,k] = A1[i,0] * B1[0,k]
    for i in range(n):  # i = 0, 1
        for k in range(2):  # k = 0, 1 (these correspond to global columns 3, 4)
            global_k = 3 + k  # columns 3 and 4
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + 0] = 1.0  # j=0, the "1" part of the 3
            v_vec[0 * p + global_k] = 1.0  # j=0, global column
            w_vec[global_k * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)
    
    # A2*B2: 2x2x2 Strassen
    # A2 is 2x2 (rows of A with j=1,2), B2 is 2x2 (rows of B with j=1,2, columns 3,4)
    # Strassen for 2x2x2: 7 multiplications
    # Let A2 = [[a11, a12], [a21, a22]] where a_ij corresponds to A[i, j+1] (j=0,1 -> global j=1,2)
    # Let B2 = [[b11, b12], [b21, b22]] where b_ij corresponds to B[j+1, global_k] (j=0,1 -> global j=1,2)
    # C2 = A2 * B2
    # Strassen:
    # M1 = (a21+a22) * (b11)          -> wait, let me use standard Strassen
    # Actually for 2x2:
    # M1 = (a21 + a22) * b11
    # M2 = (a21 + a22) * (b21 + b22)  -> no
    # Let me use the standard formulation:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (c+d)(e+f)
    # M2 = (a+b)h
    # M3 = a(e-g)
    # M4 = d(f-h)
    # M5 = (a+d)(e+h)
    # M6 = (b-d)(g+h)
    # M7 = (c-a)(e+f)
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M5
    # C22 = M1 - M3 + M4 + M7
    
    # Here a=A2[0,0]=A[0,1], b=A2[0,1]=A[0,2], c=A2[1,0]=A[1,1], d=A2[1,1]=A[1,2]
    # e=B2[0,0]=B[1,3], f=B2[0,1]=B[1,4], g=B2[1,0]=B[2,3], h=B2[1,1]=B[2,4]
    
    # Each M is a product of linear combinations, so each M is a sum of rank-1 terms.
    # For M1 = (c+d)(e+f): this is (A[1,1]+A[1,2]) * (B[1,3]+B[1,4])
    # This contributes to C21 and C22.
    
    # Let's define helper to add a product of two linear combos
    # u_combo: list of (j, coeff) for A row i... wait, we need to be careful.
    
    # For each Strassen multiplication, we have a linear combo of A entries and a linear combo of B entries.
    # The A entries are A[i, j] for specific i, j. The B entries are B[j, global_k] for specific j, global_k.
    # The product gives a contribution to C[i, global_k].
    
    # M1 = (A[1,1]+A[1,2]) * (B[1,3]+B[1,4])
    # This is a single rank-1 term in the tensor? No, it's a product of two linear combos,
    # which expands to 4 rank-1 terms. But in the tensor decomposition, we can keep it as one
    # "virtual" multiplication, but for the explicit U,V,W we need to expand.
    
    # Actually, for the explicit decomposition, each M corresponds to one rank-1 term in the
    # bilinear algorithm. The U vector has the coefficients for the A combo, V for B combo,
    # and W is determined by where the result goes.
    
    # For M1 = (c+d)(e+f), c=A[1,1], d=A[1,2], e=B[1,3], f=B[1,4]
    # U: A[1,1] has coeff 1, A[1,2] has coeff 1 -> U[1*3+1]=1, U[1*3+2]=1
    # V: B[1,3] has coeff 1, B[1,4] has coeff 1 -> V[1*5+3]=1, V[1*5+4]=1
    # W: M1 contributes to C[1,3] with +1, C[1,4] with +1, C[0,3] with -1, C[0,4] with -1
    # Wait, M1 = (c+d)(e+f) contributes to C22 with +1 and C11 with -1 (from M1 - M3 + M4 + M7 for C22... let me recheck)
    
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M5
    # C22 = M1 - M3 + M4 + M7
    
    # So M1 contributes: C12 += M1, C22 += M1
    # M2 contributes: C12 += M2, C11 -= M2
    # M3 contributes: C21 += M3, C22 -= M3
    # M4 contributes: C11 += M4, C22 += M4
    # M5 contributes: C11 += M5, C21 += M5
    # M6 contributes: C11 += M6
    # M7 contributes: C22 += M7
    
    # Let me define the 7 Strassen terms:
    # Each term: (A_combo, B_combo, W_contribs)
    # A_combo: dict {global_j: coeff} for a fixed i? No, the A combo can span multiple i.
    # Actually, in Strassen for 2x2x2, each M involves entries from both rows of A2 and both rows of B2.
    
    # Let me be more careful. A2 is 2x2 with entries A2[r,s] = A[r, s+1] for r=0,1 and s=0,1 (global j = s+1)
    # B2 is 2x2 with entries B2[r,s] = B[r+1, 3+s] for r=0,1 and s=0,1 (global j = r+1, global_k = 3+s)
    
    # M1 = (c+d)(e+f) = (A[1,1]+A[1,2])(B[1,3]+B[1,4])
    # A part: A[1,1] coeff 1, A[1,2] coeff 1
    # B part: B[1,3] coeff 1, B[1,4] coeff 1
    # W: C[0,4] += M1 (since C12, i=0, k=4->global_k=4), C[1,4] += M1 (C22, i=1, k=4)
    
    # M2 = (a+b)h = (A[0,1]+A[0,2])B[2,4]
    # A part: A[0,1] coeff 1, A[0,2] coeff 1
    # B part: B[2,4] coeff 1
    # W: C[0,4] += M2 (C12), C[0,3] -= M2 (C11, with - sign)
    
    # M3 = a(e-g) = A[0,1](B[1,3]-B[2,3])
    # A part: A[0,1] coeff 1
    # B part: B[1,3] coeff 1, B[2,3] coeff -1
    # W: C[1,3] += M3 (C21), C[1,4] -= M3 (C22, with - sign)
    
    # M4 = d(f-h) = A[1,2](B[1,4]-B[2,4])
    # A part: A[1,2] coeff 1
    # B part: B[1,4] coeff 1, B[2,4] coeff -1
    # W: C[0,3] += M4 (C11), C[1,4] += M4 (C22)
    
    # M5 = (a+d)(e+h) = (A[0,1]+A[1,2])(B[1,3]+B[2,4])
    # A part: A[0,1] coeff 1, A[1,2] coeff 1
    # B part: B[1,3] coeff 1, B[2,4] coeff 1
    # W: C[0,3] += M5 (C11), C[1,3] += M5 (C21)
    
    # M6 = (b-d)(g+h) = (A[0,2]-A[1,2])(B[2,3]+B[2,4])
    # A part: A[0,2] coeff 1, A[1,2] coeff -1
    # B part: B[2,3] coeff 1, B[2,4] coeff 1
    # W: C[0,3] += M6 (C11)
    
    # M7 = (c-a)(e+f) = (A[1,1]-A[0,1])(B[1,3]+B[1,4])
    # A part: A[1,1] coeff 1, A[0,1] coeff -1
    # B part: B[1,3] coeff 1, B[1,4] coeff 1
    # W: C[1,4] += M7 (C22)
    
    strassen_terms = [
        # (A_coefficients as list of (global_i, global_j, coeff), B_coefficients as list of (global_j, global_k, coeff), W_coefficients as list of (global_k, global_i, coeff))
        ([ (1,1,1.0), (1,2,1.0) ], [ (1,3,1.0), (1,4,1.0) ], [ (4,0,1.0), (4,1,1.0) ]),  # M1
        ([ (0,1,1.0), (0,2,1.0) ], [ (2,4,1.0) ], [ (4,0,1.0), (3,0,-1.0) ]),  # M2
        ([ (0,1,1.0) ], [ (1,3,1.0), (2,3,-1.0) ], [ (3,1,1.0), (4,1,-1.0) ]),  # M3
        ([ (1,2,1.0) ], [ (1,4,1.0), (2,4,-1.0) ], [ (3,0,1.0), (4,1,1.0) ]),  # M4
        ([ (0,1,1.0), (1,2,1.0) ], [ (1,3,1.0), (2,4,1.0) ], [ (3,0,1.0), (3,1,1.0) ]),  # M5
        ([ (0,2,1.0), (1,2,-1.0) ], [ (2,3,1.0), (2,4,1.0) ], [ (3,0,1.0) ]),  # M6
        ([ (1,1,1.0), (0,1,-1.0) ], [ (1,3,1.0), (1,4,1.0) ], [ (4,1,1.0) ]),  # M7
    ]
    
    for a_coeffs, b_coeffs, w_coeffs in strassen_terms:
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        for (gi, gj, coeff) in a_coeffs:
            u_vec[gi * m + gj] += coeff
        for (gj, gk, coeff) in b_coeffs:
            v_vec[gj * p + gk] += coeff
        for (gk, gi, coeff) in w_coeffs:
            w_vec[gk * n + gi] += coeff
        add_rank1(u_vec, v_vec, w_vec)
    
    # Now build U, V, W as 2D arrays
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
