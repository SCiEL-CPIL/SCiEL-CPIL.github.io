DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Strategy: split p=7 into p1=3 and p2=4.
    # Split m=4 into two blocks of size 2.
    #
    # Part 1: multiply (2 x 4) by (4 x 3) -> (2 x 3).
    #   Split m=4 into m1=2, m2=2.
    #   Sub-problem A: (2 x 2) x (2 x 3) -> (2 x 3). Rank 12 (schoolbook).
    #   Sub-problem B: (2 x 2) x (2 x 3) -> (2 x 3). Rank 12 (schoolbook).
    #   Total rank for Part 1: 24.
    #
    # Part 2: multiply (2 x 4) by (4 x 4) -> (2 x 4).
    #   Split m=4 into m1=2, m2=2.
    #   Split p=4 into p1=2, p2=2.
    #   Sub-problem C: (2 x 2) x (2 x 2) -> (2 x 2). Rank 7 (Strassen).
    #   Sub-problem D: (2 x 2) x (2 x 2) -> (2 x 2). Rank 7 (Strassen).
    #   Sub-problem E: (2 x 2) x (2 x 2) -> (2 x 2). Rank 7 (Strassen).
    #   Sub-problem F: (2 x 2) x (2 x 2) -> (2 x 2). Rank 7 (Strassen).
    #   Total rank for Part 2: 4 * 7 = 28.
    #
    # Total rank: 24 + 28 = 52.

    rank_total = 52
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # We'll build a helper to add a sub-decomposition at a given offset.
    def add_sub(U_sub, V_sub, W_sub, i_off, j_off, k_off, rank_off):
        # U_sub shape: (n_sub * m_sub, rank_sub)
        # V_sub shape: (m_sub * p_sub, rank_sub)
        # W_sub shape: (p_sub * n_sub, rank_sub)
        n_sub = len(U_sub) // len(U_sub[0]) if False else None
        # Actually we need to know the dimensions to map indices.
        # Let's pass dimensions explicitly.
        pass

    # Better approach: build each sub-problem explicitly and accumulate.

    # --- Part 1: (2x4) x (4x3) -> (2x3) ---
    # j in [0..3], k in [0..2]
    # j = 2*a + b, a in [0,1], b in [0,1]
    # For a=0 (j=0,1), b=0->j=0, b=1->j=1
    # For a=1 (j=2,3), b=0->j=2, b=1->j=3
    rank_off = 0
    for a in range(2):
        # Sub-problem: (2x2) x (2x3) -> (2x3), rank 12
        # Local indices: i in [0,1], b in [0,1], k in [0,2]
        # Global j = 2*a + b
        for i in range(2):
            for b in range(2):
                for k in range(3):
                    U[i * 4 + (2*a + b)][rank_off] = 1
                    V[(2*a + b) * 7 + k][rank_off] = 1
                    W[k * 2 + i][rank_off] = 1
                    rank_off += 1
    # rank_off should be 24

    # --- Part 2: (2x4) x (4x4) -> (2x4) ---
    # j in [0..3], k in [3..6]
    # j = 2*a + b, a in [0,1], b in [0,1]
    # k_local = k - 3, k_local in [0..3]
    # k_local = 2*c + d, c in [0,1], d in [0,1]
    # So c=0->k=3,4; c=1->k=5,6
    for a in range(2):
        for c in range(2):
            # Sub-problem: (2x2) x (2x2) -> (2x2), Strassen rank 7
            # Local: i in [0,1], b in [0,1], d in [0,1]
            # Global j = 2*a + b
            # Global k = 3 + 2*c + d
            # Strassen formulas:
            # P1 = (A11+A22)(B11+B22)
            # P2 = (A21+A22)B11
            # P3 = A11(B12-B22)
            # P4 = A22(B21-B11)
            # P5 = (A11+A12)B22
            # P6 = (A21-A11)(B11+B12)
            # P7 = (A12-A22)(B21+B22)
            #
            # A is 2x2: A[i][b], i in {0,1}, b in {0,1}
            # B is 2x2: B[b][d], b in {0,1}, d in {0,1}
            # C is 2x2: C[i][d]
            #
            # U[i*m+b, col] corresponds to A[i][b]
            # V[b*p+d, col] corresponds to B[b][d]
            # W[d*n+i, col] corresponds to C[d][i]  (note: W[k*n+i] -> W[d*n+i])

            # P1
            for i in [0, 1]:
                for b in [0, 1]:
                    # A[i][b] contributes to P1 if (i==0 and b==0) or (i==1 and b==1)
                    if (i == 0 and b == 0) or (i == 1 and b == 1):
                        U[i*4 + (2*a+b)][rank_off] += 1
            for b in [0, 1]:
                for d in [0, 1]:
                    # B[b][d] contributes to P1 if (b==0 and d==0) or (b==1 and d==1)
                    if (b == 0 and d == 0) or (b == 1 and d == 1):
                        V[(2*a+b)*7 + (3+2*c+d)][rank_off] += 1
            for d in [0, 1]:
                for i in [0, 1]:
                    # C[d][i] = P1 if (d==0,i==0),(d==0,i==1),(d==1,i==0),(d==1,i==1) all 1
                    W[d*2 + i][rank_off] += 1
            rank_off += 1

            # P2: (A21+A22)B11 -> C21, C22
            # A21: i=1,b=0; A22: i=1,b=1
            for i in [1]:
                for b in [0, 1]:
                    U[i*4 + (2*a+b)][rank_off] += 1
            # B11: b=0,d=0
            V[(2*a+0)*7 + (3+2*c+0)][rank_off] += 1
            # C: i=1, d=0 and d=1
            for d in [0, 1]:
                W[d*2 + 1][rank_off] += 1
            rank_off += 1

            # P3: A11(B12-B22) -> C12, -C22
            # A11: i=0,b=0
            U[0*4 + (2*a+0)][rank_off] += 1
            # B12: b=0,d=1; B22: b=1,d=1
            V[(2*a+0)*7 + (3+2*c+1)][rank_off] += 1
            V[(2*a+1)*7 + (3+2*c+1)][rank_off] -= 1
            # C: C12 = +1, C22 = -1
            # d=1, i=0 -> +1; d=1, i=1 -> -1
            W[1*2 + 0][rank_off] += 1
            W[1*2 + 1][rank_off] -= 1
            rank_off += 1

            # P4: A22(B21-B11) -> C21, -C11
            # A22: i=1,b=1
            U[1*4 + (2*a+1)][rank_off] += 1
            # B21: b=1,d=0; B11: b=0,d=0
            V[(2*a+1)*7 + (3+2*c+0)][rank_off] += 1
            V[(2*a+0)*7 + (3+2*c+0)][rank_off] -= 1
            # C: C21 = +1, C11 = -1
            # d=0, i=1 -> +1; d=0, i=0 -> -1
            W[0*2 + 1][rank_off] += 1
            W[0*2 + 0][rank_off] -= 1
            rank_off += 1

            # P5: (A11+A12)B22 -> C12, C22
            # A11: i=0,b=0; A12: i=0,b=1
            for b in [0, 1]:
                U[0*4 + (2*a+b)][rank_off] += 1
            # B22: b=1,d=1
            V[(2*a+1)*7 + (3+2*c+1)][rank_off] += 1
            # C: C12=+1, C22=+1
            W[1*2 + 0][rank_off] += 1
            W[1*2 + 1][rank_off] += 1
            rank_off += 1

            # P6: (A21-A11)(B11+B12) -> C21, C22
            # A21: i=1,b=0 -> +1; A11: i=0,b=0 -> -1
            U[1*4 + (2*a+0)][rank_off] += 1
            U[0*4 + (2*a+0)][rank_off] -= 1
            # B11: b=0,d=0; B12: b=0,d=1
            V[(2*a+0)*7 + (3+2*c+0)][rank_off] += 1
            V[(2*a+0)*7 + (3+2*c+1)][rank_off] += 1
            # C: C21=+1, C22=+1
            W[0*2 + 1][rank_off] += 1
            W[1*2 + 1][rank_off] += 1
            rank_off += 1

            # P7: (A12-A22)(B21+B22) -> C12, C22
            # A12: i=0,b=1 -> +1; A22: i=1,b=1 -> -1
            U[0*4 + (2*a+1)][rank_off] += 1
            U[1*4 + (2*a+1)][rank_off] -= 1
            # B21: b=1,d=0; B22: b=1,d=1
            V[(2*a+1)*7 + (3+2*c+0)][rank_off] += 1
            V[(2*a+1)*7 + (3+2*c+1)][rank_off] += 1
            # C: C12=+1, C22=+1
            W[1*2 + 0][rank_off] += 1
            W[1*2 + 1][rank_off] += 1
            rank_off += 1

    return {"U": U, "V": V, "W": W}
