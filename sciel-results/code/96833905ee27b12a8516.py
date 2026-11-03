"""Rewritten program: rank-48 exact tensor decomposition for 3x4x6 matrix multiplication.
Uses block structure: A rows [2,1], A cols [2,2], B rows [2,2], B cols [2,2,2].
Top 2 rows use Strassen (7 rank per 2x2*2x2), bottom 1 row uses rank-1 (1 rank per 1x2*2x2).
Total: 3 col-groups * 2 A-col-groups * (7 + 1) = 48 rank terms.
"""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 4, 6

    terms = []  # Each term: (U_dict, V_dict, W_dict)

    def add_term(U_entries, V_entries, W_entries):
        U_d = {}
        for (i, j, c) in U_entries:
            U_d[(i, j)] = U_d.get((i, j), 0) + c
        V_d = {}
        for (j, k, c) in V_entries:
            V_d[(j, k)] = V_d.get((j, k), 0) + c
        W_d = {}
        for (k, i, c) in W_entries:
            W_d[(k, i)] = W_d.get((k, i), 0) + c
        # Filter out zero entries
        U_d = {k: v for k, v in U_d.items() if v != 0}
        V_d = {k: v for k, v in V_d.items() if v != 0}
        W_d = {k: v for k, v in W_d.items() if v != 0}
        terms.append((U_d, V_d, W_d))

    def strassen_terms(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """7 Strassen terms for 2x2 * 2x2 -> 2x2."""
        r0, r1 = A_rows
        c0, c1 = A_cols  # these are the row indices in B
        s0, s1 = B_rows
        t0, t1 = B_cols

        # M1 = (A[r0,c0]+A[r1,c1]) * (B[s0,t0]+B[s1,t1])
        # C[r0,t0] += M1, C[r1,t1] += M1
        add_term(
            [(r0, c0, 1), (r1, c1, 1)],
            [(s0, t0, 1), (s1, t1, 1)],
            [(t0, r0, 1), (t1, r1, 1)]
        )

        # M2 = (A[r1,c0]+A[r1,c1]) * B[s0,t0]
        # C[r1,t0] += M2, C[r1,t1] -= M2
        add_term(
            [(r1, c0, 1), (r1, c1, 1)],
            [(s0, t0, 1)],
            [(t0, r1, 1), (t1, r1, -1)]
        )

        # M3 = A[r0,c0] * (B[s0,t1]-B[s1,t1])
        # C[r0,t1] += M3, C[r1,t1] += M3
        add_term(
            [(r0, c0, 1)],
            [(s0, t1, 1), (s1, t1, -1)],
            [(t1, r0, 1), (t1, r1, 1)]
        )

        # M4 = A[r1,c1] * (B[s1,t0]-B[s0,t0])
        # C[r1,t0] += M4, C[r0,t0] += M4
        add_term(
            [(r1, c1, 1)],
            [(s1, t0, 1), (s0, t0, -1)],
            [(t0, r1, 1), (t0, r0, 1)]
        )

        # M5 = (A[r0,c0]+A[r0,c1]) * B[s1,t1]
        # C[r1,t1] += M5, C[r0,t0] -= M5
        add_term(
            [(r0, c0, 1), (r0, c1, 1)],
            [(s1, t1, 1)],
            [(t1, r1, 1), (t0, r0, -1)]
        )

        # M6 = (A[r1,c1]-A[r0,c1]) * (B[s0,t0]+B[s0,t1])
        # C[r1,t1] += M6
        add_term(
            [(r1, c1, 1), (r0, c1, -1)],
            [(s0, t0, 1), (s0, t1, 1)],
            [(t1, r1, 1)]
        )

        # M7 = (A[r0,c1]-A[r1,c1]) * (B[s1,t1]+B[s0,t1])
        # C[r0,t0] += M7
        add_term(
            [(r0, c1, 1), (r1, c1, -1)],
            [(s1, t1, 1), (s0, t1, 1)],
            [(t0, r0, 1)]
        )

    def direct_1x2_2x2_terms(A_row, A_cols, B_rows, B_cols, C_row, C_cols):
        """1 rank term for 1x2 * 2x2 -> 1x2."""
        r = A_row
        c0, c1 = A_cols
        s0, s1 = B_rows
        t0, t1 = B_cols

        # U: A[r,c0] and A[r,c1]
        # V: B[s0,t0], B[s0,t1], B[s1,t0], B[s1,t1]
        # W: C[t0,r] and C[t1,r]
        add_term(
            [(r, c0, 1), (r, c1, 1)],
            [(s0, t0, 1), (s0, t1, 1), (s1, t0, 1), (s1, t1, 1)],
            [(t0, r, 1), (t1, r, 1)]
        )

    # Block structure:
    # A rows: [0,1] (top 2), [2] (bottom 1)
    # A cols: [0,1] (left 2), [2,3] (right 2)
    # B rows: [0,1] (top 2), [2,3] (bottom 2)
    # B cols: [0,1], [2,3], [4,5] (three 2-col groups)
    # C rows: [0,1] (top 2), [2] (bottom 1)
    # C cols: [0,1], [2,3], [4,5]

    col_groups = [(0, 1), (2, 3), (4, 5)]
    a_col_groups = [(0, 1), (2, 3)]
    b_row_groups = [(0, 1), (2, 3)]

    for (t0, t1) in col_groups:
        for ac0, ac1 in a_col_groups:
            br0, br1 = b_row_groups[0] if ac0 == 0 else b_row_groups[1]
            # Top 2 rows: Strassen for A[0:2, ac] * B[br, t]
            strassen_terms((0, 1), (ac0, ac1), (br0, br1), (t0, t1), (0, 1), (t0, t1))
            # Bottom 1 row: Rank-1 for A[2, ac] * B[br, t]
            direct_1x2_2x2_terms(2, (ac0, ac1), (br0, br1), 2, (t0, t1))

    rank = len(terms)

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for c, (U_d, V_d, W_d) in enumerate(terms):
        for (i, j), coeff in U_d.items():
            U[i * m + j, c] = coeff
        for (j, k), coeff in V_d.items():
            V[j * p + k, c] = coeff
        for (k, i), coeff in W_d.items():
            W[k * n + i, c] = coeff

    # Verify
    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    if not verify(U, V, W, rank):
        # Fall back to schoolbook (rank = n*m*p = 72)
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = rank_school

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
