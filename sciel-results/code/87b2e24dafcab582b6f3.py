"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, r):
        """Check if U,V,W of rank r correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T_reconstructed = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    # Strassen-based block decomposition
    def build_strassen_decomposition():
        terms = []

        def add_term(U_entries, V_entries, W_entries):
            U_d = {}
            for (i, j, c) in U_entries:
                U_d[(i, j)] = c
            V_d = {}
            for (j, k, c) in V_entries:
                V_d[(j, k)] = c
            W_d = {}
            for (k, i, c) in W_entries:
                W_d[(k, i)] = c
            terms.append((U_d, V_d, W_d))

        def strassen_terms(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
            r0, r1 = A_rows
            c0, c1 = A_cols
            s0, s1 = B_rows
            t0, t1 = B_cols

            # M1 = (A[r0,c0]+A[r1,c1]) * (B[s0,t0]+B[s1,t1])
            add_term(
                [(r0, c0, 1), (r1, c1, 1)],
                [(s0, t0, 1), (s1, t1, 1)],
                [(t0, r0, 1), (t1, r1, 1)]
            )

            # M2 = (A[r1,c0]+A[r1,c1]) * B[s0,t0]
            add_term(
                [(r1, c0, 1), (r1, c1, 1)],
                [(s0, t0, 1)],
                [(t0, r1, 1), (t1, r1, -1)]
            )

            # M3 = A[r0,c0] * (B[s0,t1]-B[s1,t1])
            add_term(
                [(r0, c0, 1)],
                [(s0, t1, 1), (s1, t1, -1)],
                [(t1, r0, 1), (t1, r1, 1)]
            )

            # M4 = A[r1,c1] * (B[s1,t0]-B[s0,t0])
            add_term(
                [(r1, c1, 1)],
                [(s1, t0, 1), (s0, t0, -1)],
                [(t0, r1, 1), (t0, r0, 1)]
            )

            # M5 = (A[r0,c0]+A[r0,c1]) * B[s1,t1]
            add_term(
                [(r0, c0, 1), (r0, c1, 1)],
                [(s1, t1, 1)],
                [(t1, r1, 1), (t0, r0, -1)]
            )

            # M6 = (A[r1,c1]-A[r0,c1]) * (B[s0,t0]+B[s0,t1])
            add_term(
                [(r1, c1, 1), (r0, c1, -1)],
                [(s0, t0, 1), (s0, t1, 1)],
                [(t1, r1, 1)]
            )

            # M7 = (A[r0,c1]-A[r1,c1]) * (B[s1,t1]+B[s0,t1])
            add_term(
                [(r0, c1, 1), (r1, c1, -1)],
                [(s1, t1, 1), (s0, t1, 1)],
                [(t0, r0, 1)]
            )

        def direct_1x2_2x2_terms(A_row, A_cols, B_rows, B_cols, C_row, C_cols):
            r = A_row
            c0, c1 = A_cols
            s0, s1 = B_rows
            t0, t1 = B_cols

            add_term([(r, c0, 1)], [(s0, t0, 1)], [(t0, r, 1)])
            add_term([(r, c1, 1)], [(s1, t0, 1)], [(t0, r, 1)])
            add_term([(r, c0, 1)], [(s0, t1, 1)], [(t1, r, 1)])
            add_term([(r, c1, 1)], [(s1, t1, 1)], [(t1, r, 1)])

        # Block structure for 3x4 * 4x6
        col_groups = [(0, 1), (2, 3), (4, 5)]

        for ci, (t0, t1) in enumerate(col_groups):
            # Top 2 rows: C[0:2, t0:t1+1] = A[0:2, 0:2]*B[0:2, t0:t1+1] + A[0:2, 2:4]*B[2:4, t0:t1+1]
            strassen_terms((0, 1), (0, 1), (0, 1), (t0, t1), (0, 1), (t0, t1))
            strassen_terms((0, 1), (2, 3), (2, 3), (t0, t1), (0, 1), (t0, t1))

            # Bottom 1 row: C[2, t0:t1+1] = A[2, 0:2]*B[0:2, t0:t1+1] + A[2, 2:4]*B[2:4, t0:t1+1]
            direct_1x2_2x2_terms(2, (0, 1), (0, 1), (t0, t1), 2, (t0, t1))
            direct_1x2_2x2_terms(2, (2, 3), (2, 3), (t0, t1), 2, (t0, t1))

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

        return U, V, W, rank

    # Try Strassen-based decomposition
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    if time.time() < deadline - 20:
        U_str, V_str, W_str, r_str = build_strassen_decomposition()
        if verify(U_str, V_str, W_str, r_str):
            best_rank = r_str
            best_U, best_V, best_W = U_str, V_str, W_str

    # Try to reduce rank further by looking for linear dependencies
    if time.time() < deadline - 10 and best_rank < rank_school:
        # Try random perturbations to see if we can reduce rank
        np.random.seed(42)
        for r_try in range(max(24, best_rank - 5), best_rank):
            if time.time() > deadline - 5:
                break
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                U_try = np.random.randn(n * m, r_try)
                V_try = np.random.randn(m * p, r_try)
                W_try = np.random.randn(p * n, r_try)
                if verify(U_try, V_try, W_try, r_try):
                    best_rank = r_try
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break

    # Convert to half-integers if possible
    def to_half_integer(arr):
        arr = np.array(arr, dtype=float)
        rounded = np.round(arr * 2) / 2
        return rounded

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    # Verify the rounded version
    if not verify(U_out, V_out, W_out, best_rank):
        # Fall back to original
        U_out = best_U
        V_out = best_V
        W_out = best_W
        if not verify(U_out, V_out, W_out, best_rank):
            # Fall back to schoolbook
            U_out = U_school
            V_out = V_school
            W_out = W_school
            best_rank = rank_school

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
