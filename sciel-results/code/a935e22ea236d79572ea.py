def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = [0]

    def add_strassen_block(A_indices, B_indices, C_indices):
        # A_indices: 4 entries [(i,j),...] local 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
        # B_indices: 4 entries [(j,k),...] local 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
        # C_indices: 4 entries [(i,k),...] local 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
        mks = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
        ]
        for A_c, B_c, C_c in mks:
            for la, c in A_c.items():
                gi, gj = A_indices[la]
                U[gi * m + gj][col[0]] += c
            for lb, c in B_c.items():
                gj, gk = B_indices[lb]
                V[gj * p + gk][col[0]] += c
            for lc, c in C_c.items():
                gi, gk = C_indices[lc]
                W[gk * n + gi][col[0]] += c
            col[0] += 1

    def add_schoolbook_2x2x1(A_indices, B_indices, C_indices):
        # A: 2x2 local 0..3, B: 2x1 local 0=(0,0),1=(1,0), C: 2x1 local 0=(0,0),1=(1,0)
        mults = [
            (0, 0, 0), (1, 1, 0),   # C00 = A00*B00 + A01*B10
            (2, 0, 1), (3, 1, 1),   # C10 = A10*B00 + A11*B10
        ]
        for la, lb, lc in mults:
            gi, gj = A_indices[la]
            U[gi * m + gj][col[0]] = 1
            gj2, gk = B_indices[lb]
            V[gj2 * p + gk][col[0]] = 1
            gi2, gk2 = C_indices[lc]
            W[gk2 * n + gi2][col[0]] = 1
            col[0] += 1

    A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]

    for A_block, B_row in [(A_L, (0, 1)), (A_R, (2, 3))]:
        r0, r1 = B_row
        # Cols 0-1 (2x2)
        B2 = [(r0, 0), (r0, 1), (r1, 0), (r1, 1)]
        C2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
        add_strassen_block(A_block, B2, C2)
        # Cols 2-3 (2x2)
        B2 = [(r0, 2), (r0, 3), (r1, 2), (r1, 3)]
        C2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
        add_strassen_block(A_block, B2, C2)
        # Cols 4-5 (2x2)
        B2 = [(r0, 4), (r0, 5), (r1, 4), (r1, 5)]
        C2 = [(0, 4), (0, 5), (1, 4), (1, 5)]
        add_strassen_block(A_block, B2, C2)
        # Col 6 (2x1)
        B1 = [(r0, 6), (r1, 6)]
        C1 = [(0, 6), (1, 6)]
        add_schoolbook_2x2x1(A_block, B1, C1)

    return {"U": U, "V": V, "W": W}
