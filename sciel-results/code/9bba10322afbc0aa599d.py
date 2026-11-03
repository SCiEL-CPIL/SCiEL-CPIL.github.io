"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import sys

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_rank66():
        U66 = np.zeros((n * m, 66))
        V66 = np.zeros((m * p, 66))
        W66 = np.zeros((p * n, 66))
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U66[A00, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1
            W66[C10, col] += 1; W66[C11, col] -= 1
            col += 1

            U66[A00, col] += 1
            V66[B01, col] += 1; V66[B11, col] -= 1
            W66[C01, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A11, col] += 1
            V66[B10, col] += 1; V66[B00, col] -= 1
            W66[C00, col] += 1; W66[C10, col] += 1
            col += 1

            U66[A00, col] += 1; U66[A01, col] += 1
            V66[B11, col] += 1
            W66[C00, col] -= 1; W66[C01, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A00, col] -= 1
            V66[B00, col] += 1; V66[B01, col] += 1
            W66[C11, col] += 1
            col += 1

            U66[A01, col] += 1; U66[A11, col] -= 1
            V66[B10, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U66[A0, col] += 1; V66[B00, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B10, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A0, col] += 1; V66[B01, col] += 1; W66[C1, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B11, col] += 1; W66[C1, col] += 1
            col += 1

        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        return U66, V66, W66

    def gf11_peel(U66, V66, W66, r):
        """Try to find linear dependencies over GF(11) and reduce rank."""
        q = 11
        # Build the (n*m*p) x r matrix M over GF(11)
        # M[ijk, c] = U66[i*m+j, c] * V66[j*p+k, c] * W66[k*n+i, c] mod q
        M = np.zeros((n * m * p, r), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    flat_idx = i * m * p + j * p + k
                    for c in range(r):
                        M[flat_idx, c] = (U66[i * m + j, c] * V66[j * p + k, c] * W66[k * n + i, c]) % q

        # Gaussian elimination over GF(11) to find rank and null space
        M_work = M.copy()
        rows, cols = M_work.shape
        rank = 0
        pivot_cols = []
        pivot_rows = []

        for c in range(cols):
            if rank >= rows:
                break
            # Find pivot
            pivot = -1
            for r_idx in range(rank, rows):
                if M_work[r_idx, c] != 0:
                    pivot = r_idx
                    break
            if pivot == -1:
                continue
            # Swap
            if pivot != rank:
                M_work[[rank, pivot]] = M_work[[pivot, rank]]
            # Normalize
            inv = pow(int(M_work[rank, c]), q - 2, q)
            M_work[rank, :] = (M_work[rank, :] * inv) % q
            # Eliminate
            for r_idx in range(rows):
                if r_idx != rank and M_work[r_idx, c] != 0:
                    factor = M_work[r_idx, c]
                    M_work[r_idx, :] = (M_work[r_idx, :] - factor * M_work[rank, :]) % q
            pivot_cols.append(c)
            pivot_rows.append(rank)
            rank += 1

        if rank < r:
            # Find null space basis
            # Free columns are those not in pivot_cols
            free_cols = [c for c in range(cols) if c not in pivot_cols]
            null_basis = []
            for fc in free_cols:
                vec = np.zeros(cols, dtype=np.int64)
                vec[fc] = 1
                for i, pc in enumerate(pivot_cols):
                    vec[pc] = (-M_work[pivot_rows[i], fc]) % q
                null_basis.append(vec)

            # Try to eliminate one column at a time
            # For each null vector, we can express one column as combination of others
            # Try removing each free column
            for fc in free_cols:
                # Check if removing column fc gives a valid decomposition
                keep = [c for c in range(r) if c != fc]
                U_new = U66[:, keep]
                V_new = V66[:, keep]
                W_new = W66[:, keep]
                if verify(U_new, V_new, W_new, r - 1):
                    return U_new, V_new, W_new, r - 1

        return U66, V66, W66, r

    def try_rank_reduction(U66, V66, W66, r):
        """Try to reduce rank by finding a linear dependency."""
        # First try GF(11) peeling
        U_new, V_new, W_new, r_new = gf11_peel(U66, V66, W66, r)
        if r_new < r:
            return U_new, V_new, W_new, r_new

        # Try rational null space
        # Build M matrix over rationals
        M = np.zeros((n * m * p, r), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    flat_idx = i * m * p + j * p + k
                    for c in range(r):
                        M[flat_idx, c] = U66[i * m + j, c] * V66[j * p + k, c] * W66[k * n + i, c]

        try:
            # Find null space
            U_sv, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-9
            rank_M = np.sum(s > tol)
            if rank_M < r:
                # Null space is last (r - rank_M) columns of Vt
                null_vecs = Vt[rank_M:, :]
                for nv in null_vecs:
                    # Try removing the column with largest absolute value in null vector
                    idx_to_remove = np.argmax(np.abs(nv))
                    if abs(nv[idx_to_remove]) < 1e-9:
                        continue
                    keep = [c for c in range(r) if c != idx_to_remove]
                    U_new = U66[:, keep]
                    V_new = V66[:, keep]
                    W_new = W66[:, keep]
                    if verify(U_new, V_new, W_new, r - 1):
                        return U_new, V_new, W_new, r - 1
        except:
            pass

        return U66, V66, W66, r

    def schoolbook():
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
        return U, V, W, rank_school

    # Build initial rank-66 decomposition
    U66, V66, W66 = build_rank66()

    if not verify(U66, V66, W66, 66):
        U, V, W, r = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    U_best, V_best, W_best, r_best = U66.copy(), V66.copy(), W66.copy(), 66

    # Try GF(11) peeling
    U_try, V_try, W_try, r_try = gf11_peel(U66, V66, W66, 66)
    if r_try < r_best:
        U_best, V_best, W_best, r_best = U_try, V_try, W_try, r_try

    # Try rational null space
    U_try, V_try, W_try, r_try = try_rank_reduction(U66, V66, W66, 66)
    if r_try < r_best:
        U_best, V_best, W_best, r_best = U_try, V_try, W_try, r_try

    # Try to further reduce
    if r_best < 66:
        U_try, V_try, W_try, r_try = gf11_peel(U_best, V_best, W_best, r_best)
        if r_try < r_best:
            U_best, V_best, W_best, r_best = U_try, V_try, W_try, r_try

    if r_best > 72:
        U, V, W, r = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
