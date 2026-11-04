"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Uses block decomposition (Strassen) with fast vectorized verification.
Attempts to find rank < 84 via structured search and local refinement.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    ii_arr = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj_arr = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk_arr = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    U_idx_base = ii_arr * m + jj_arr
    V_idx_base = jj_arr * p + kk_arr
    W_idx_base = kk_arr * n + ii_arr

    def verify_decomposition(U, V, W):
        U_sel = U[U_idx_base]
        V_sel = V[V_idx_base]
        W_sel = W[W_idx_base]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        all_half_int = (np.allclose(U * 2, np.round(U * 2)) and
                       np.allclose(V * 2, np.round(V * 2)) and
                       np.allclose(W * 2, np.round(W * 2)))
        return all_half_int

    def add_rank1_term(U_vecs, V_vecs, W_vecs, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_strassen_2x2x2(U_vecs, V_vecs, W_vecs, i_off, j_off, k_off):
        i0 = i_off; i1 = i_off + 1
        j0 = j_off; j1 = j_off + 1
        k0 = k_off; k1 = k_off + 1
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0], [1],
                      [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0], [1],
                      [j0*p+k1, j1*p+k1], [1, -1],
                      [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j1], [1],
                      [j1*p+k0, j0*p+k0], [1, -1],
                      [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0, i0*m+j1], [1, 1],
                      [j1*p+k1], [1],
                      [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j0, i0*m+j0], [1, -1],
                      [j0*p+k0, j0*p+k1], [1, 1],
                      [k1*n+i1], [1])
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j1, i1*m+j1], [1, -1],
                      [j1*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0], [1])

    def gen_schoolbook(U_vecs, V_vecs, W_vecs, n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term(U_vecs, V_vecs, W_vecs,
                                  [ii*m+jj], [1.0],
                                  [jj*p+kk], [1.0],
                                  [kk*n+ii], [1.0])

    def build_block_decomp(rows_split, cols_split):
        U_vecs = []
        V_vecs = []
        W_vecs = []
        row_offsets = [0]
        for rs in rows_split:
            row_offsets.append(row_offsets[-1] + rs)
        col_offsets = [0]
        for cs in cols_split:
            col_offsets.append(col_offsets[-1] + cs)
        for r_idx in range(len(rows_split)):
            r_start = row_offsets[r_idx]
            r_size = rows_split[r_idx]
            for c_idx in range(len(cols_split)):
                c_start = col_offsets[c_idx]
                c_size = cols_split[c_idx]
                if r_size == 1:
                    gen_schoolbook(U_vecs, V_vecs, W_vecs, 1, m, c_size, r_start, 0, c_start)
                elif r_size == 2:
                    for m_start in range(0, m, 2):
                        m_end = min(m_start + 2, m)
                        m_size = m_end - m_start
                        if m_size == 2:
                            for k_start in range(c_start, c_start + c_size, 2):
                                k_end = min(k_start + 2, c_start + c_size)
                                k_size = k_end - k_start
                                if k_size == 2:
                                    gen_strassen_2x2x2(U_vecs, V_vecs, W_vecs, r_start, m_start, k_start)
                                elif k_size == 1:
                                    gen_schoolbook(U_vecs, V_vecs, W_vecs, 2, 2, 1, r_start, m_start, k_start)
                        else:
                            for k_start in range(c_start, c_start + c_size):
                                gen_schoolbook(U_vecs, V_vecs, W_vecs, 2, 1, 1, r_start, m_start, k_start)
                else:
                    gen_schoolbook(U_vecs, V_vecs, W_vecs, r_size, m, c_size, r_start, 0, c_start)
        if not U_vecs:
            return None
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        if not verify_decomposition(U, V, W):
            return None
        rank = U.shape[1]
        return {"U": (np.round(U * 2) / 2).tolist(),
                "V": (np.round(V * 2) / 2).tolist(),
                "W": (np.round(W * 2) / 2).tolist(),
                "rank": rank}

    def try_reduce_rank(U_mat, V_mat, W_mat, target_rank, time_budget=30):
        """Try to reduce rank by removing terms and adjusting remaining ones."""
        R = U_mat.shape[1]
        if R <= target_rank:
            return None
        start_t = time.time()
        # Try each term for removal
        for r_to_remove in range(R):
            if time.time() - start_t > time_budget:
                break
            # Create modified matrices without term r_to_remove
            mask = np.ones(R, dtype=bool)
            mask[r_to_remove] = False
            U_r = U_mat[:, mask].copy()
            V_r = V_mat[:, mask].copy()
            W_r = W_mat[:, mask].copy()
            R_new = R - 1
            # Check if current solution still works (it won't in general)
            # Instead, try to find small adjustments
            # Build the residual
            U_sel = U_r[U_idx_base]
            V_sel = V_r[V_idx_base]
            W_sel = W_r[W_idx_base]
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            residual = 1.0 - vals
            if np.max(np.abs(residual)) < 1e-9:
                # Already valid!
                if verify_decomposition(U_r, V_r, W_r):
                    return {"U": (np.round(U_r * 2) / 2).tolist(),
                            "V": (np.round(V_r * 2) / 2).tolist(),
                            "W": (np.round(W_r * 2) / 2).tolist(),
                            "rank": R_new}
            # Try iterative refinement: adjust each remaining term slightly
            # Use a simple gradient-free approach: try perturbing each term
            for attempt in range(50):
                if time.time() - start_t > time_budget:
                    break
                # Pick a random term to adjust
                r_adj = np.random.randint(0, R_new)
                # Try small perturbations in U, V, W for this term
                for dim, mat in enumerate([U_r, V_r, W_r]):
                    for coord in np.where(np.abs(mat[:, r_adj]) > 0.1)[0]:
                        for delta in [0.5, -0.5, 1.0, -1.0]:
                            mat_old = mat[coord, r_adj]
                            mat[coord, r_adj] = mat_old + delta
                            U_sel = U_r[U_idx_base]
                            V_sel = V_r[V_idx_base]
                            W_sel = W_r[W_idx_base]
                            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
                            if np.allclose(vals, 1.0, atol=1e-9):
                                if verify_decomposition(U_r, V_r, W_r):
                                    return {"U": (np.round(U_r * 2) / 2).tolist(),
                                            "V": (np.round(V_r * 2) / 2).tolist(),
                                            "W": (np.round(W_r * 2) / 2).tolist(),
                                            "rank": R_new}
                            mat[coord, r_adj] = mat_old
        return None

    row_split_options = [[2, 1], [1, 2], [3], [1, 1, 1]]
    col_split_options = [[7], [2, 5], [5, 2], [3, 4], [4, 3],
                         [1, 6], [6, 1], [2, 2, 3], [2, 3, 2], [3, 2, 2]]

    best_rank = float('inf')
    best_result = None

    for rows_split in row_split_options:
        if sum(rows_split) != n:
            continue
        for cols_split in col_split_options:
            if sum(cols_split) != p:
                continue
            if time.time() > deadline - 40:
                break
            result = build_block_decomp(rows_split, cols_split)
            if result and result["rank"] < best_rank:
                best_rank = result["rank"]
                best_result = result
        if time.time() > deadline - 40:
            break

    # Try to reduce rank from best result
    if best_result and time.time() < deadline - 10:
        U_mat = np.array(best_result["U"])
        V_mat = np.array(best_result["V"])
        W_mat = np.array(best_result["W"])
        reduced = try_reduce_rank(U_mat, V_mat, W_mat, best_rank - 1,
                                  time_budget=min(30, deadline - 10 - time.time()))
        if reduced and reduced["rank"] < best_rank:
            best_rank = reduced["rank"]
            best_result = reduced

    if best_result:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

    # Fallback: schoolbook
    U_vecs_fb, V_vecs_fb, W_vecs_fb = [], [], []
    gen_schoolbook(U_vecs_fb, V_vecs_fb, W_vecs_fb, n, m, p, 0, 0, 0)
    U_fb = np.array(U_vecs_fb).T
    V_fb = np.array(V_vecs_fb).T
    W_fb = np.array(W_vecs_fb).T
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
