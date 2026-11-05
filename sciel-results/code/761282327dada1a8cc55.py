"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: start from a verified rank-33 Strassen-based block decomposition,
    then try to reduce the rank by (a) structured Fourier-diagonalized local search
    that exploits the block structure, and (b) column-elimination / absorption moves.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin for 150s wall-clock

    n, m, p = DIMENSIONS
    dim_a = n * m   # 6
    dim_b = m * p   # 18
    dim_c = p * n   # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Fast exact verification: sum_r U[:,r] * V[:,r] * W[:,r] == T
    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.array_equal(result, T)

    def verify_float(U, V, W, tol=1e-9):
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
        for c in range(rank):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, T.astype(np.float64), atol=tol, rtol=0)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Rank-33 decomposition: 3 blocks. Each block = Strassen 2x2x2 (7 terms)
        on the first two columns of B plus a Standard 2x1x2 (4 terms) on the third.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)

        S_U = [
            [[1, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]],
            [[1, 1], [0, 0]], [[-1, 0], [1, 0]], [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]], [[1, 0], [0, 0]], [[0, 1], [0, -1]], [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]], [[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]
        ]
        Std_U = [[[1], [0]], [[1], [0]], [[0], [1]], [[0], [1]]]
        Std_V = [[[1, 0]], [[0, 1]], [[1, 0]], [[0, 1]]]
        Std_W = [
            [[1, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 1]]
        ]

        col = 0
        for b in range(3):
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        V[j * p + (2 * b + kp), col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        W[(2 * b + kp) * n + i, col] = S_W[r][kp][i]
                col += 1
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    V[2 * p + (2 * b + kp), col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        W[(2 * b + kp) * n + i, col] = Std_W[r][kp][i]
                col += 1
        return U, V, W

    # Baseline
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    if not verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Lower bound from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    lower_bound = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2),
                      np.linalg.matrix_rank(T3))

    # ---- Fourier change of basis on the shared middle dimension (j) ----
    # DFT over j=0,1,2. Change of basis in V (rows) and U (rows) keeps the
    # tensor form valid because the basis changes are invertible and exact.
    def dft_basis():
        # P[j, j'] such that new_j = sum_j' P[j,j'] old_j', P = DFT (1/sqrt3)
        P = np.zeros((m, m), dtype=np.float64)
        for j in range(m):
            for jp in range(m):
                P[j, jp] = np.exp(2j * np.pi * j * jp / m) / np.sqrt(m)
        return P

    P = dft_basis()
    # New U rows: U'[i, j'] = sum_j U[i,j] P[j', j]  ->  U' = (P.T @ U_reshaped)
    # New V rows: V'[j', k] = sum_j P[j', j] V[j, k] ->  V' = (P @ V_reshaped)
    def to_fourier(U, V, W):
        Ur = U.reshape(n, m, best_rank)
        Vr = V.reshape(m, p, best_rank)
        Uf = np.einsum('jj', P.T, Ur)  # placeholder, do explicitly
        Uf = np.zeros((n, m, best_rank), dtype=np.float64)
        Vf = np.zeros((m, p, best_rank), dtype=np.float64)
        for i in range(n):
            for jp in range(m):
                for r in range(best_rank):
                    Uf[i, jp, r] = sum(P[jp, j] * Ur[i, j, r] for j in range(m))
        for jp in range(m):
            for k in range(p):
                for r in range(best_rank):
                    Vf[jp, k, r] = sum(P[jp, j] * Vr[j, k, r] for j in range(m))
        return (Uf.reshape(dim_a, best_rank),
                Vf.reshape(dim_b, best_rank),
                W.astype(np.float64))

    # Try Fourier-diagonalized local search: perturb within the transformed
    # basis and attempt to drop a term, then transform back and verify exactly.
    def fourier_local_search():
        global best_U, best_V, best_W, best_rank
        if time.time() > deadline - 8:
            return
        Uf, Vf, Wf = to_fourier(best_U, best_V, best_W)
        # Try to reduce rank by one: drop a column and re-fit the dropped
        # contribution onto the remaining columns via least squares on the
        # residual, then round to small integers and verify.
        for drop in range(best_rank):
            if time.time() > deadline - 8:
                return
            mask = np.ones(best_rank, dtype=bool)
            mask[drop] = False
            U_rem = Uf[:, mask].copy()
            V_rem = Vf[:, mask].copy()
            W_rem = Wf[:, mask].copy()
            # residual tensor in Fourier space
            res = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
            for c in range(U_rem.shape[1]):
                res += np.einsum('i,j,k->ijk', U_rem[:, c], V_rem[:, c], W_rem[:, c])
            # target in Fourier space
            Tf = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
            Tfr = Tf.reshape(n, m, p)
            for i in range(n):
                for k in range(p):
                    for jp in range(m):
                        Tfr[i, jp, k] = sum(P[jp, j] * T.reshape(n, m, p)[i, j, k] for j in range(m))
            diff = Tf - res
            # try a single rank-1 term to absorb diff: U_new[:,drop], V_new, W_new
            # Use SVD of diff matricization to get a rank-1 approximation.
            diff_mat = diff.reshape(dim_a, dim_b * dim_c)
            try:
                Usv, _, Vsv = np.linalg.svd(diff_mat, full_matrices=False)
            except Exception:
                continue
            u1 = Usv[:, 0]
            rest = Vsv[0, :].reshape(dim_b, dim_c)
            try:
                Usv2, _, Vsv2 = np.linalg.svd(rest, full_matrices=False)
            except Exception:
                continue
            v1 = Usv2[:, 0]
            w1 = Vsv2[0, :]
            # Round to nearest integer (small values expected)
            u1i = np.rint(u1).astype(np.int64)
            v1i = np.rint(v1).astype(np.int64)
            w1i = np.rint(w1).astype(np.int64)
            # Build candidate integer tensors
            U_cand = np.zeros((dim_a, best_rank - 1), dtype=np.int64)
            V_cand = np.zeros((dim_b, best_rank - 1), dtype=np.int64)
            W_cand = np.zeros((dim_c, best_rank - 1), dtype=np.int64)
            # round the remaining columns too
            U_cand = np.rint(U_rem).astype(np.int64)
            V_cand = np.rint(V_rem).astype(np.int64)
            W_cand = np.rint(W_rem).astype(np.int64)
            # append the new rank-1 term
            U_cand = np.column_stack([U_cand, u1i])
            V_cand = np.column_stack([V_cand, v1i])
            W_cand = np.column_stack([W_cand, w1i])
            if verify_decomposition(U_cand, V_cand, W_cand):
                best_U, best_V, best_W = U_cand, V_cand, W_cand
                best_rank = best_rank - 1
                return

    # ---- Column-elimination / absorption local search ----
    def column_elimination_search():
        global best_U, best_V, best_W, best_rank
        improved = True
        while improved and best_rank > lower_bound:
            improved = False
            if time.time() > deadline - 6:
                break
            # Try dropping each column
            for drop in range(best_rank):
                if time.time() > deadline - 6:
                    break
                mask = np.ones(best_rank, dtype=bool)
                mask[drop] = False
                if verify_decomposition(best_U[:, mask], best_V[:, mask], best_W[:, mask]):
                    best_U = best_U[:, mask]
                    best_V = best_V[:, mask]
                    best_W = best_W[:, mask]
                    best_rank -= 1
                    improved = True
                    break
            # Try absorbing column d into column e: for each entry, move
            # contribution of d into e by adding d's vector scaled, then zero d.
            # Only valid if d is a scalar multiple of e in all three factors.
            if not improved and time.time() > deadline - 6:
                for d in range(best_rank):
                    for e in range(best_rank):
                        if d == e:
                            continue
                        if time.time() > deadline - 6:
                            break
                        # find scalar s.t. U[:,d] = s*U[:,e], etc.
                        def scalar_match(a, b):
                            nz = np.nonzero(b)[0]
                            if len(nz) == 0:
                                return 0.0 if np.all(a == 0) else None
                            s = a[nz[0]] / b[nz[0]]
                            if not np.allclose(a, s * b, atol=1e-9):
                                return None
                            return s
                        su = scalar_match(best_U[:, d], best_U[:, e])
                        if su is None:
                            continue
                        sv = scalar_match(best_V[:, d], best_V[:, e])
                        if sv is None or abs(sv - su) > 1e-9:
                            continue
                        sw = scalar_match(best_W[:, d], best_W[:, e])
                        if sw is None or abs(sw - su) > 1e-9:
                            continue
                        # absorb: e += s*d, then drop d
                        best_U[:, e] += su * best_U[:, d]
                        best_V[:, e] += su * best_V[:, d]
                        best_W[:, e] += su * best_W[:, d]
                        mask = np.ones(best_rank, dtype=bool)
                        mask[d] = False
                        if verify_decomposition(best_U[:, mask], best_V[:, mask], best_W[:, mask]):
                            best_U = best_U[:, mask]
                            best_V = best_V[:, mask]
                            best_W = best_W[:, mask]
                            best_rank -= 1
                            improved = True
                            break
                    if improved:
                        break

    # Run searches
    fourier_local_search()
    column_elimination_search()

    # Final safety: ensure we still have a valid decomposition
    if not verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = strassen_block_decomposition()
        best_rank = 33
        if not verify_decomposition(best_U, best_V, best_W):
            best_U, best_V, best_W = schoolbook()
            best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
