import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115

    n, m, p = 4, 4, 4

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen 2x2 block terms: (A_comb, B_comb, C_comb)
    block_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]

    sub_terms = block_terms

    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)

    rank_idx = 0
    for A_comb, B_comb, C_comb in block_terms:
        for k in range(7):
            A_loc, B_loc, C_loc = sub_terms[k]

            U_elems = {}
            for (bi, bj), coeff_block in A_comb:
                for (lr, lc), coeff_loc in A_loc:
                    gi = 2 * bi + lr
                    gj = 2 * bj + lc
                    idx = u_idx(gi, gj)
                    U_elems[idx] = U_elems.get(idx, 0) + coeff_block * coeff_loc

            V_elems = {}
            for (bi, bj), coeff_block in B_comb:
                for (lr, lc), coeff_loc in B_loc:
                    gj = 2 * bi + lr
                    gk = 2 * bj + lc
                    idx = v_idx(gj, gk)
                    V_elems[idx] = V_elems.get(idx, 0) + coeff_block * coeff_loc

            W_elems = {}
            for (ci, cj), coeff_block in C_comb:
                for (lr, lc), coeff_loc in C_loc:
                    gi = 2 * ci + lr
                    gj = 2 * cj + lc
                    idx = w_idx(gj, gi)
                    W_elems[idx] = W_elems.get(idx, 0) + coeff_block * coeff_loc

            for idx, val in U_elems.items():
                U[idx, rank_idx] += val
            for idx, val in V_elems.items():
                V[idx, rank_idx] += val
            for idx, val in W_elems.items():
                W[idx, rank_idx] += val

            rank_idx += 1

    def verify_fast(U_mat, V_mat, W_mat, tol=1e-10):
        U_t = np.asarray(U_mat, dtype=float)
        V_t = np.asarray(V_mat, dtype=float)
        W_t = np.asarray(W_mat, dtype=float)
        r = U_t.shape[1]
        if r == 0:
            return False
        U_r = U_t.reshape(n, m, r)
        V_r = V_t.reshape(m, p, r)
        W_r = W_t.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, 1.0, atol=tol)

    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if not verify_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the 64x49 tensor matrix for null-space analysis
    terms_flat = np.zeros((64, 49), dtype=float)
    for k in range(49):
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    terms_flat[i * 16 + j * 4 + l, k] = (
                        U[i * m + j, k] * V[j * p + l, k] * W[l * n + i, k]
                    )

    # Find null-space vector
    try:
        U_svd, s, Vt_svd = np.linalg.svd(terms_flat, full_matrices=True)
        tol_sv = max(terms_flat.shape) * np.finfo(float).eps * (s[0] if s[0] > 0 else 1.0)
        num_sv = len(s)
        null_dim = sum(1 for idx in range(num_sv) if s[idx] < tol_sv)

        if null_dim > 0:
            c = Vt_svd[num_sv - null_dim, :].copy()
            k_elim = int(np.argmax(np.abs(c)))
            if abs(c[k_elim]) > 1e-10:
                c = c / c[k_elim]
                for j in range(49):
                    if j != k_elim:
                        U[:, j] *= (1.0 - c[j])

                keep = [j for j in range(49) if j != k_elim]
                U48 = U[:, keep].copy()
                V48 = V[:, keep].copy()
                W48 = W[:, keep].copy()

                if verify_fast(U48, V48, W48):
                    if time.time() - start_time > DEADLINE:
                        return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}

                    # Try to find another dependency in the 48-term decomposition
                    terms48 = np.zeros((64, 48), dtype=float)
                    for k2 in range(48):
                        for i in range(n):
                            for j in range(m):
                                for l in range(p):
                                    terms48[i * 16 + j * 4 + l, k2] = (
                                        U48[i * m + j, k2] * V48[j * p + l, k2] * W48[l * n + i, k2]
                                    )

                    try:
                        U_svd2, s2, Vt_svd2 = np.linalg.svd(terms48, full_matrices=True)
                        tol_sv2 = max(terms48.shape) * np.finfo(float).eps * (s2[0] if s2[0] > 0 else 1.0)
                        num_sv2 = len(s2)
                        null_dim2 = sum(1 for idx in range(num_sv2) if s2[idx] < tol_sv2)

                        if null_dim2 > 0:
                            c2 = Vt_svd2[num_sv2 - null_dim2, :].copy()
                            k_elim2 = int(np.argmax(np.abs(c2)))
                            if abs(c2[k_elim2]) > 1e-10:
                                c2 = c2 / c2[k_elim2]
                                for j2 in range(48):
                                    if j2 != k_elim2:
                                        U48[:, j2] *= (1.0 - c2[j2])

                                keep2 = [j2 for j2 in range(48) if j2 != k_elim2]
                                U47 = U48[:, keep2].copy()
                                V47 = V48[:, keep2].copy()
                                W47 = W48[:, keep2].copy()

                                if verify_fast(U47, V47, W47):
                                    if time.time() - start_time > DEADLINE:
                                        return {"U": U47.tolist(), "V": V47.tolist(), "W": W47.tolist()}

                                    # Try a third reduction
                                    terms47 = np.zeros((64, 47), dtype=float)
                                    for k3 in range(47):
                                        for i in range(n):
                                            for j in range(m):
                                                for l in range(p):
                                                    terms47[i * 16 + j * 4 + l, k3] = (
                                                        U47[i * m + j, k3] * V47[j * p + l, k3] * W47[l * n + i, k3]
                                                    )
                                    try:
                                        U_svd3, s3, Vt_svd3 = np.linalg.svd(terms47, full_matrices=True)
                                        tol_sv3 = max(terms47.shape) * np.finfo(float).eps * (s3[0] if s3[0] > 0 else 1.0)
                                        num_sv3 = len(s3)
                                        null_dim3 = sum(1 for idx in range(num_sv3) if s3[idx] < tol_sv3)
                                        if null_dim3 > 0:
                                            c3 = Vt_svd3[num_sv3 - null_dim3, :].copy()
                                            k_elim3 = int(np.argmax(np.abs(c3)))
                                            if abs(c3[k_elim3]) > 1e-10:
                                                c3 = c3 / c3[k_elim3]
                                                for j3 in range(47):
                                                    if j3 != k_elim3:
                                                        U47[:, j3] *= (1.0 - c3[j3])
                                                keep3 = [j3 for j3 in range(47) if j3 != k_elim3]
                                                U46 = U47[:, keep3].copy()
                                                V46 = V47[:, keep3].copy()
                                                W46 = W47[:, keep3].copy()
                                                if verify_fast(U46, V46, W46):
                                                    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
                                    except Exception:
                                        pass

                                    return {"U": U47.tolist(), "V": V47.tolist(), "W": W47.tolist()}
                    except Exception:
                        pass

                    return {"U": U48.tolist(), "V": V48.tolist(), "W": W48.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
