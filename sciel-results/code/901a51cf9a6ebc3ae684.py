"""Matrix multiplication 4,4,8 via two independent Strassen rank-49 4x4x4 blocks."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    def get_strassen_2x2():
        # U, V, W for 2x2x2 (rank 7).
        # U: row i*2+j (4 rows), V: row j*2+k (4 rows), W: row k*2+i (4 rows)
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W

    def strassen_4x4x4():
        # Build rank-49 decomposition for 4x4 x 4x4 -> 4x4.
        # U: 16 rows (i*4+j), V: 16 rows (j*4+k), W: 16 rows (k*4+i)
        U2, V2, W2 = get_strassen_2x2()
        R = 49
        U4 = [[0] * R for _ in range(16)]
        V4 = [[0] * R for _ in range(16)]
        W4 = [[0] * R for _ in range(16)]

        # 7 super terms of the 4x4 Strassen. Each is (sum of A 2x2 blocks) * (sum of B 2x2 blocks).
        # Block coords (I,J) in {0,1}^2; local 2x2 index l = il*2+jl, global = (I*2+il)*4 + (J*2+jl).
        def add_term(A_blocks, B_blocks, W_blocks, sign):
            # A_blocks: list of (I,J); B_blocks: list of (K,L); W_blocks: list of (K,I) for C.
            # The super term is (sum_IJ A[I,J]) * (sum_KL B[K,L]) -> contributes to C blocks W[K,I].
            # We decompose the 2x2 product of (sum of A blocks) and (sum of B blocks) using rank-7.
            # For each of the 7 local Strassen terms s, the local A-side linear combo and
            # local B-side linear combo are scaled and added to the corresponding global blocks.
            for s in range(7):
                col = 0  # will compute below
            # Instead iterate: for each local Strassen slot s, for each A block and B block,
            # the contribution is U2[local_a, s] * V2[local_b, s] * W2[local_c, s] mapped to globals.
            for s in range(7):
                # Collect (local_a, coeff_a), (local_b, coeff_b), (local_c, coeff_c)
                a_pairs = []
                for (I, J) in A_blocks:
                    for il in range(2):
                        for jl in range(2):
                            la = il * 2 + jl
                            c = U2[la][s]
                            if c:
                                a_pairs.append((I, J, il, jl, c))
                b_pairs = []
                for (K, L) in B_blocks:
                    for jl in range(2):
                        for kl in range(2):
                            lb = jl * 2 + kl
                            c = V2[lb][s]
                            if c:
                                b_pairs.append((K, L, jl, kl, c))
                w_pairs = []
                for (K, I) in W_blocks:
                    for kl in range(2):
                        for il in range(2):
                            lc = kl * 2 + il
                            c = W2[lc][s]
                            if c:
                                w_pairs.append((K, I, kl, il, c))
                if not (a_pairs and b_pairs and w_pairs):
                    continue
                # Each triple (a, b, w) gives a rank-1 term in the overall tensor.
                # But the 7 local slots already form a rank-7 block; we must add them as
                # 7 separate global columns. For slot s, we add one global column per
                # (a, b, w) triple? No - the local rank-7 decomposition means slot s is a
                # single rank-1 in the 2x2x2 tensor. So for the super term, slot s is a
                # single rank-1 over the *summed* blocks. The summed blocks mean:
                # U_side = sum over a_blocks of the local U pattern, i.e. for each global (I,J,il,jl)
                # with local la, coefficient c_a. Similarly V and W.
                # So slot s of the super term is ONE global rank-1 term whose U vector is
                # the sum of U2[la,s] over the A blocks, V vector sum over B blocks, W over W blocks.
                # That is a single column in U4/V4/W4.
                # Assign column index = s (0..6) within this super term; we'll offset later.
                yield s, a_pairs, b_pairs, w_pairs

        # Define the 7 super terms: (A_blocks, B_blocks, W_blocks)
        super_terms = [
            # T1: (A00+A11)(B00+B11) -> C00, C11
            ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]),
            # T2: (A10+A11)B00 -> C10, C11
            ([(1, 0), (1, 1)], [(0, 0)], [(0, 1), (1, 1)]),
            # T3: A00(B01-B11) -> C01, C11
            ([(0, 0)], [(0, 1), (1, 1)], [(1, 0), (1, 1)]),
            # T4: A11(B10-B00) -> C00, C10
            ([(1, 1)], [(1, 0), (0, 0)], [(0, 1), (1, 1)]),
            # T5: (A00+A01)B11 -> C00, C01
            ([(0, 0), (0, 1)], [(1, 1)], [(0, 0), (0, 1)]),
            # T6: (A10-A00)(B00+B01) -> C11
            ([(1, 0), (0, 0)], [(0, 0), (0, 1)], [(1, 1)]),
            # T7: (A01-A11)(B10+B11) -> C00, C11
            ([(0, 1), (1, 1)], [(1, 0), (1, 1)], [(0, 0), (1, 1)]),
        ]

        col = 0
        for (A_blocks, B_blocks, W_blocks) in super_terms:
            for s in range(7):
                a_pairs = []
                for (I, J) in A_blocks:
                    for il in range(2):
                        for jl in range(2):
                            la = il * 2 + jl
                            c = U2[la][s]
                            if c:
                                a_pairs.append(((I * 2 + il) * 4 + (J * 2 + jl), c))
                b_pairs = []
                for (K, L) in B_blocks:
                    for jl in range(2):
                        for kl in range(2):
                            lb = jl * 2 + kl
                            c = V2[lb][s]
                            if c:
                                b_pairs.append(((K * 2 + jl) * 4 + (L * 2 + kl), c))
                w_pairs = []
                for (K, I) in W_blocks:
                    for kl in range(2):
                        for il in range(2):
                            lc = kl * 2 + il
                            c = W2[lc][s]
                            if c:
                                w_pairs.append(((K * 2 + kl) * 4 + (I * 2 + il), c))
                for (row, c) in a_pairs:
                    U4[row][col] += c
                for (row, c) in b_pairs:
                    V4[row][col] += c
                for (row, c) in w_pairs:
                    W4[row][col] += c
                col += 1
        return U4, V4, W4

    U4, V4, W4 = strassen_4x4x4()

    # Now build 4x4x8 as two 4x4x4 blocks: B columns 0..3 and 4..7.
    # U is shared (same A), V is split into two halves, W is split into two halves.
    R = 98
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(32)]
    W = [[0] * R for _ in range(32)]

    # Left half: B cols 0..3, W cols 0..3
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            V[r][c] = V4[r][c]
            W[r][c] = W4[r][c]
    # Right half: B cols 4..7 (V rows offset by 16), W cols 4..7 (W rows offset by 16)
    for r in range(16):
        for c in range(49):
            U[r][49 + c] = U4[r][c]
            V[r + 16][49 + c] = V4[r][c]
            W[r + 16][49 + c] = W4[r][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
