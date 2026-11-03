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

    def verify_fast(U, V, W, r):
        """Fast vectorized check if U,V,W of rank r correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape to (n, m, p, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=1e-9)

    def verify(U, V, W, r):
        """Check if U,V,W of rank r correctly decompose the tensor."""
        return verify_fast(U, V, W, r)

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

    if time.time() < deadline - 30:
        U_str, V_str, W_str, r_str = build_strassen_decomposition()
        if verify(U_str, V_str, W_str, r_str):
            best_rank = r_str
            best_U, best_V, best_W = U_str, V_str, W_str

    # Local search: try to merge or drop terms to reduce rank
    if time.time() < deadline - 10 and best_rank < rank_school:
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        r_cur = best_rank

        # Try dropping individual terms
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            if r_cur <= 1:
                break
            # Try dropping each term
            dropped = False
            for c in range(r_cur):
                if time.time() > deadline - 10:
                    break
                # Create new matrices without term c
                mask = [x for x in range(r_cur) if x != c]
                U_new = U_cur[:, mask]
                V_new = V_cur[:, mask]
                W_new = W_cur[:, mask]
                if verify_fast(U_new, V_new, W_new, r_cur - 1):
                    U_cur = U_new
                    V_cur = V_new
                    W_cur = W_new
                    r_cur -= 1
                    best_rank = r_cur
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    dropped = True
                    break
            if not dropped:
                break

        # Try merging pairs of terms (replace two terms with one linear combination)
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            if r_cur <= 1:
                break
            merged = False
            for c1 in range(r_cur):
                if time.time() > deadline - 5:
                    break
                for c2 in range(c1 + 1, r_cur):
                    if time.time() > deadline - 5:
                        break
                    # Try to find alpha, beta such that term_c1 + alpha*term_c2 can be replaced
                    # Actually, try: new_term = term_c1 + t * term_c2, then drop term_c2
                    # This is complex; instead try random linear combinations
                    for t in [-2, -1, -0.5, 0.5, 1, 2]:
                        if time.time() > deadline - 5:
                            break
                        # Create new matrices: replace col c1 with col c1 + t*col c2, drop col c2
                        U_new = U_cur.copy()
                        V_new = V_cur.copy()
                        W_new = W_cur.copy()
                        U_new[:, c1] = U_cur[:, c1] + t * U_cur[:, c2]
                        V_new[:, c1] = V_cur[:, c1] + t * V_cur[:, c2]
                        W_new[:, c1] = W_cur[:, c1] + t * W_cur[:, c2]
                        # Now drop column c2
                        mask = [x for x in range(r_cur) if x != c2]
                        U_final = U_new[:, mask]
                        V_final = V_new[:, mask]
                        W_final = W_new[:, mask]
                        if verify_fast(U_final, V_final, W_final, r_cur - 1):
                            U_cur = U_final
                            V_cur = V_final
                            W_cur = W_final
                            r_cur -= 1
                            best_rank = r_cur
                            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if not merged:
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
