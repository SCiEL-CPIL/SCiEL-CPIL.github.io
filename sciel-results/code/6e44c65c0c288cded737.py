"""Optimized tensor decomposition for 2x3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = 2, 3, 6
    # Split m=3 into m1=2, m2=1
    # A is 2x3: A = [A1 | A2], A1 is 2x2, A2 is 2x1
    # B is 3x6: B = [B1; B2], B1 is 2x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x6, split B1 into 3 blocks of 2x2
    # Use Strassen (rank 7) for each 2x2x2 block
    # A2*B2: 2x1 * 1x6, schoolbook rank 12
    
    U = []
    V = []
    W = []
    
    # Strassen for 2x2x2:
    # C = A*B where A,B,C are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # Indices: A[i,j] for i in {0,1}, j in {0,1}
    # B[k,l] for k in {0,1}, l in {0,1}
    # C[i,l] for i in {0,1}, l in {0,1}
    
    # For A1*B1 block b (b=0,1,2), B1 block is B[0:2, 2*b:2*b+2]
    # U index for A: i*2 + j (i in 0,1; j in 0,1) -> 0,1,2,3
    # V index for B: k*6 + (2*b+l) (k in 0,1; l in 0,1) -> for block b: 2*b, 2*b+1, 6+2*b, 6+2*b+1
    # W index for C: l*n + i = l*2 + i (i in 0,1; l in 0,1) -> 0,1,2,3
    
    strassen_defs = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is dict: index -> coefficient
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 2: 1}, {0: 1, 2: 1, 1: -1, 3: -1}),  # C00 += M1, C11 += -M1... wait
    ]
    
    # Let me be more careful. For each Strassen product Mx, it contributes to specific C entries.
    # I'll build the full decomposition.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add a rank-1 term. u_coeffs: dict A_idx->coeff, v_coeffs: dict B_idx->coeff, w_coeffs: dict C_idx->coeff"""
        max_u = max(u_coeffs.keys()) + 1 if u_coeffs else 0
        max_v = max(v_coeffs.keys()) + 1 if v_coeffs else 0
        max_w = max(w_coeffs.keys()) + 1 if w_coeffs else 0
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        for idx, c in u_coeffs.items():
            u_vec[idx] = c
        for idx, c in v_coeffs.items():
            v_vec[idx] = c
        for idx, c in w_coeffs.items():
            w_vec[idx] = c
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # A1*B1: 2x2 * 2x6, B1 split into 3 blocks of 2x2
    # A1 indices: A[i, j] -> U index i*m + j = i*3 + j, but A1 is A[0:2, 0:2], so U indices 0,1,2,3
    # B1 indices: B[k, l] for k in 0,1; l in 0..5. V index k*p + l = k*6 + l
    # C indices: C[i, l] -> W index l*n + i = l*2 + i
    
    # Strassen definitions for 2x2x2:
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices (local): 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += -M1? No:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So each Mx contributes to specific C entries with specific signs.
    
    strassen_mults = [
        # (name, u_coeffs, v_local_coeffs, c_contribs)
        # c_contribs: dict C_local_idx -> coeff
        ("M1", {0: 1, 3: 1}, {0: 1, 2: 1}, {0: 1, 3: 1}),
        ("M2", {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ("M3", {0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        ("M4", {3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        ("M5", {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ("M6", {2: 1, 0: -1}, {0: 1, 1: 1}, {}),  # M6 not used in final? 
        ("M7", {1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: 1}),
    ]
    
    # Wait, M6 is not in the final C formula. Let me recheck Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # So M6 is not used! That means Strassen is rank 7 but one of the 7 products is not needed?
    # No, all 7 are used. Let me recheck.
    
    # Standard Strassen:
    # M1 = (A11 - A22)(B21 + B22)
    # M2 = (A21 + A22)B11
    # M3 = (A11 + A12)B22
    # M4 = A11(B12 - B22)
    # M5 = A22(B21 - B11)
    # M6 = (A11 + A21)B12
    # M7 = (A12 + A22)(B21 - B11)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M1 + M6 - M7
    # C22 = M1 - M2 + M3 + M4
    
    # Let me use this standard form. A is 2x2 with A[i,j], B is 2x2 with B[i,j].
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    strassen_mults = [
        # (u_coeffs, v_local_coeffs, c_contribs)
        # M1 = (A11-A22)(B21+B22)
        ({0: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 2: -1, 3: 1}),
        # M2 = (A21+A22)B11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = (A11+A12)B22
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),
        # M4 = A11(B12-B22)
        ({0: 1}, {1: 1, 3: -1}, {0: 1, 3: 1}),
        # M5 = A22(B21-B11)
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 1: 1}),
        # M6 = (A11+A21)B12
        ({0: 1, 2: 1}, {1: 1}, {2: 1}),
        # M7 = (A12+A22)(B21-B11)
        ({1: 1, 3: 1}, {2: 1, 0: -1}, {0: 1, 2: -1}),
    ]
    
    # For block b (0,1,2), B1 block is B[0:2, 2*b:2*b+2]
    # B local indices: 0=B[0,2*b], 1=B[0,2*b+1], 2=B[1,2*b], 3=B[1,2*b+1]
    # V global indices: 0*6+2*b=2*b, 0*6+2*b+1=2*b+1, 1*6+2*b=6+2*b, 1*6+2*b+1=6+2*b+1
    
    for b in range(3):
        for (u_coeffs, v_local, c_contribs) in strassen_mults:
            # Map v_local to global V indices
            v_global = {}
            for local_idx, coeff in v_local.items():
                # local 0 -> B[0, 2*b], local 1 -> B[0, 2*b+1]
                # local 2 -> B[1, 2*b], local 3 -> B[1, 2*b+1]
                if local_idx == 0:
                    gidx = 0 * 6 + (2 * b)
                elif local_idx == 1:
                    gidx = 0 * 6 + (2 * b + 1)
                elif local_idx == 2:
                    gidx = 1 * 6 + (2 * b)
                elif local_idx == 3:
                    gidx = 1 * 6 + (2 * b + 1)
                else:
                    continue
                v_global[gidx] = v_global.get(gidx, 0) + coeff
            
            # u_coeffs are for A1, which is A[0:2, 0:2], U indices 0,1,2,3
            # c_contribs are for C local indices 0,1,2,3 -> C[0,0],C[0,1],C[1,0],C[1,1]
            # C global: C[i, l] -> W index l*2 + i
            # local 0 = C[0,0] -> W[0], local 1 = C[0,1] -> W[1]
            # local 2 = C[1,0] -> W[2], local 3 = C[1,1] -> W[3]
            w_global = {}
            for local_idx, coeff in c_contribs.items():
                gidx = local_idx  # since l*2+i: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
                w_global[gidx] = w_global.get(gidx, 0) + coeff
            
            add_term(u_coeffs, v_global, w_global)
    
    # A2*B2: 2x1 * 1x6, schoolbook
    # A2 is A[0:2, 2:3], U indices: i*3+2 for i in 0,1 -> 2, 5
    # B2 is B[2:3, 0:6], V indices: 2*6+l = 12+l for l in 0..5
    # C is 2x6, W index l*2+i
    
    for i in range(2):
        for l in range(6):
            u_idx = i * 3 + 2  # A2[i, 0]
            v_idx = 2 * 6 + l  # B2[0, l]
            w_idx = l * 2 + i  # C[i, l]
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Convert to lists of lists
    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    # Verify shapes
    # U should be (n*m, rank) = (6, rank)
    # V should be (m*p, rank) = (18, rank)
    # W should be (p*n, rank) = (12, rank)
    
    return {"U": U, "V": V, "W": W}
