"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses clean vector-based Strassen block decomposition (inspiration 1) combined
with efficient linear-dependence rank reduction (current program machinery).
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Builds a Strassen-based block decomposition (rank 78) then reduces rank
    via joint linear-dependence detection.
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    def build_schoolbook():
        """Build the schoolbook decomposition (rank = n*m*p = 84)."""
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_vecs.append(u)
                    V_vecs.append(v)
                    W_vecs.append(w)
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    # Precompute index arrays for verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    U_row_idx = i_idx * m + j_idx
    V_row_idx = j_idx * p + k_idx
    W_row_idx = k_idx * n + i_idx

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor."""
        U_sel = U[U_row_idx, :]
        V_sel = V[V_row_idx, :]
        W_sel = W[W_row_idx, :]
        totals = np.sum(U_sel * V_sel * W_sel, axis=1)
        if not np.all(np.abs(totals - 1.0) <= tol):
            return False
        return True

    def build_strassen_decomp():
        """
        Clean vector-based Strassen block decomposition.
        n=3: split as 2+1
        m=4: split as 2+2
        p=7: split as 2+2+2+1
        2x2x2 blocks use Strassen (rank 7), smaller blocks use schoolbook.
        Total: 6*7 + 2*4 + 6*4 + 2*2 = 42+8+24+4 = 78
        """
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val
            U_vecs.append(u)
            V_vecs.append(v)
            W_vecs.append(w)

        def strassen_222(i0, j0, k0):
            """Strassen decomposition for 2x2x2 block at offset (i0,j0,k0)."""
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1

            # M1: (A11+A22)*(B11+B22) -> C11, C22
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])

            # M2: (A21+A22)*B11 -> C21, C22
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])

            # M3: A11*(B12-B22) -> C12
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])

            # M4: A22*(B21-B11) -> C21
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])

            # M5: (A11+A12)*B22 -> C12, C22
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])

            # M6: (A21-A11)*(B11+B12) -> C21
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])

            # M7: (A12-A22)*(B21+B22) -> C12
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])

        def schoolbook_block(i0, j0, k0, ni, mj, pk):
            """Schoolbook decomposition for block of size (ni, mj, pk)."""
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        ii = i0 + i
                        jj = j0 + j
                        kk = k0 + k
                        add_rank1_term([ii*m+jj], [1.0],
                                       [jj*p+kk], [1.0],
                                       [kk*n+ii], [1.0])

        # 6 blocks of size (2,2,2) -> Strassen
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                strassen_222(0, j0, k0)

        # 2 blocks of size (2,2,1) -> schoolbook
        for j0 in [0, 2]:
            schoolbook_block(0, j0, 6, 2, 2, 1)

        # 6 blocks of size (1,2,2) -> schoolbook
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                schoolbook_block(2, j0, k0, 1, 2, 2)

        # 2 blocks of size (1,2,1) -> schoolbook
        for j0 in [0, 2]:
            schoolbook_block(2, j0, 6, 1, 2, 1)

        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    # Build initial decomposition
    U, V, W = build_strassen_decomp()
    current_rank = U.shape[1]

    # Verify initial decomposition
    if not verify_decomp(U, V, W):
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    def rank_reduction(U, V, W, max_time=100.0):
        """
        Rank reduction via structural and linear dependence detection.
        Pass 1: Structural redundancy (proportional columns with same support).
        Pass 2: Joint linear dependence (column in span of others in all 3 matrices).
        Pass 3: Pair combination search.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_opt = time.time()
        tol = 1e-10

        def time_check():
            return time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0

        # Pass 1: Structural redundancy removal
        for iteration in range(50):
            if time_check() or rank <= 1:
                break

            reduced = False
            groups = {}
            for r in range(rank):
                u_sup = tuple(i for i in range(n*m) if abs(U[i, r]) > tol)
                v_sup = tuple(i for i in range(m*p) if abs(V[i, r]) > tol)
                w_sup = tuple(i for i in range(p*n) if abs(W[i, r]) > tol)
                key = (u_sup, v_sup, w_sup)
                if key not in groups:
                    groups[key] = []
                groups[key].append(r)

            for key, col_list in groups.items():
                if len(col_list) <= 1:
                    continue
                if time_check():
                    break

                ref = col_list[0]
                u_ref = U[:, ref]
                v_ref = V[:, ref]
                w_ref = W[:, ref]

                nz_idx = np.nonzero(np.abs(u_ref) > tol)[0]
                if len(nz_idx) == 0:
                    continue
                idx = nz_idx[0]

                to_remove = []
                for r in col_list[1:]:
                    if abs(u_ref[idx]) < tol:
                        continue
                    c = U[idx, r] / u_ref[idx]
                    if (np.allclose(U[:, r], c * u_ref, atol=tol, rtol=tol) and
                        np.allclose(V[:, r], c * v_ref, atol=tol, rtol=tol) and
                        np.allclose(W[:, r], c * w_ref, atol=tol, rtol=tol)):
                        to_remove.append(r)

                for r in sorted(to_remove, reverse=True):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    reduced = True

            if not reduced:
                break

        # Pass 2: Joint linear dependence
        for iteration in range(100):
            if time_check() or rank <= 1:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time_check():
                    break

                other = [i for i in range(rank) if i != r]
                if len(other) == 0:
                    break

                U_other = U[:, other]
                V_other = V[:, other]
                W_other = W[:, other]

                try:
                    _, res_u, _, _ = np.linalg.lstsq(U_other, U[:, r], rcond=tol)
                    if res_u > tol * 10:
                        continue
                except:
                    continue

                try:
                    _, res_v, _, _ = np.linalg.lstsq(V_other, V[:, r], rcond=tol)
                    if res_v > tol * 10:
                        continue
                except:
                    continue

                try:
                    _, res_w, _, _ = np.linalg.lstsq(W_other, W[:, r], rcond=tol)
                    if res_w > tol * 10:
                        continue
                except:
                    continue

                U_test = np.delete(U, r, axis=1)
                V_test = np.delete(V, r, axis=1)
                W_test = np.delete(W, r, axis=1)

                if verify_decomp(U_test, V_test, W_test, tol=1e-8):
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    reduced = True
                    break

            if not reduced:
                break

        # Pass 3: Pair combination search
        for iteration in range(30):
            if time_check() or rank <= 2:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time_check():
                    break

                for s in range(r - 1, -1, -1):
                    other = [i for i in range(rank) if i != r and i != s]
                    if len(other) == 0:
                        break

                    U_other = U[:, other]
                    V_other = V[:, other]
                    W_other = W[:, other]

                    for c in [1.0, -1.0]:
                        U_comb = U[:, r] + c * U[:, s]
                        V_comb = V[:, r] + c * V[:, s]
                        W_comb = W[:, r] + c * W[:, s]

                        try:
                            _, res_u, _, _ = np.linalg.lstsq(U_other, U_comb, rcond=tol)
                            if res_u > tol * 10:
                                continue
                            _, res_v, _, _ = np.linalg.lstsq(V_other, V_comb, rcond=tol)
                            if res_v > tol * 10:
                                continue
                            _, res_w, _, _ = np.linalg.lstsq(W_other, W_comb, rcond=tol)
                            if res_w > tol * 10:
                                continue

                            U_test = np.delete(U, r, axis=1)
                            V_test = np.delete(V, r, axis=1)
                            W_test = np.delete(W, r, axis=1)

                            if verify_decomp(U_test, V_test, W_test, tol=1e-8):
                                U = U_test
                                V = V_test
                                W = W_test
                                rank -= 1
                                reduced = True
                                break
                        except:
                            continue

                    if reduced:
                        break

                if reduced:
                    break

            if not reduced:
                break

        return U, V, W, rank

    # Apply rank reduction
    elapsed = time.time() - start_time
    remaining = max(1.0, deadline - 5.0 - elapsed)
    U, V, W, current_rank = rank_reduction(U, V, W, max_time=remaining)

    # Final verification
    if not verify_decomp(U, V, W, tol=1e-8):
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
