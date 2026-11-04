"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Tries multiple block decomposition strategies and returns the lowest-rank valid one.
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    best_result = None
    best_rank = float('inf')

    def try_decomposition(n_split, col_splits, middle_splits_per_col_group):
        """Try a specific decomposition strategy."""
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
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

        def gen_schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
            for i in range(n_l):
                for j in range(m_l):
                    for k in range(p_l):
                        ii = i_off + i
                        jj = j_off + j
                        kk = k_off + k
                        add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

        def gen_strassen_2x2x2(i_off, j_off, k_off):
            i0 = i_off; i1 = i_off+1
            j0 = j_off; j1 = j_off+1
            k0 = k_off; k1 = k_off+1
            
            u1 = [i0*m+j0, i1*m+j1]
            v1 = [j0*p+k0, j1*p+k1]
            w1 = [k0*n+i0, k1*n+i1]
            add_rank1_term(u1, [1, 1], v1, [1, 1], w1, [1, 1])
            
            u2 = [i1*m+j0, i1*m+j1]
            v2 = [j0*p+k0]
            w2 = [k0*n+i1, k1*n+i1]
            add_rank1_term(u2, [1, 1], v2, [1], w2, [1, -1])
            
            u3 = [i0*m+j0]
            v3 = [j0*p+k1, j1*p+k1]
            w3 = [k1*n+i0, k1*n+i1]
            add_rank1_term(u3, [1], v3, [1, -1], w3, [1, 1])
            
            u4 = [i1*m+j1]
            v4 = [j1*p+k0, j0*p+k0]
            w4 = [k0*n+i0, k0*n+i1]
            add_rank1_term(u4, [1], v4, [1, -1], w4, [1, 1])
            
            u5 = [i0*m+j0, i0*m+j1]
            v5 = [j1*p+k1]
            w5 = [k0*n+i0, k1*n+i0]
            add_rank1_term(u5, [1, 1], v5, [1], w5, [-1, 1])
            
            u6 = [i1*m+j0, i0*m+j0]
            v6 = [j0*p+k0, j0*p+k1]
            w6 = [k1*n+i1]
            add_rank1_term(u6, [1, -1], v6, [1, 1], w6, [1])
            
            u7 = [i0*m+j1, i1*m+j1]
            v7 = [j1*p+k0, j1*p+k1]
            w7 = [k0*n+i0]
            add_rank1_term(u7, [1, -1], v7, [1, 1], w7, [1])

        def gen_2x2x2x3(i_off, j_off, k_off):
            """2x2 * 2x3: Strassen for first 2 cols + schoolbook for last col. Rank 11."""
            gen_strassen_2x2x2(i_off, j_off, k_off)
            for i in range(2):
                for j in range(2):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + 2
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

        def gen_2x2x2x4(i_off, j_off, k_off):
            """2x2 * 2x4: Strassen + Strassen. Rank 14."""
            gen_strassen_2x2x2(i_off, j_off, k_off)
            gen_strassen_2x2x2(i_off, j_off, k_off + 2)

        def gen_2x2x2x7(i_off, j_off, k_off):
            """2x2 * 2x7: split as 2+2+2+1. Rank 7+7+7+4=25."""
            gen_strassen_2x2x2(i_off, j_off, k_off)
            gen_strassen_2x2x2(i_off, j_off, k_off + 2)
            gen_strassen_2x2x2(i_off, j_off, k_off + 4)
            for i in range(2):
                for j in range(2):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + 6
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

        # Build decomposition based on splits
        row_offsets = [0]
        for rs in n_split:
            row_offsets.append(row_offsets[-1] + rs)
        
        col_offsets = [0]
        for cs in col_splits:
            col_offsets.append(col_offsets[-1] + cs)
        
        for r_idx in range(len(n_split)):
            r_start = row_offsets[r_idx]
            r_size = n_split[r_idx]
            
            for c_idx in range(len(col_splits)):
                c_start = col_offsets[c_idx]
                c_size = col_splits[c_idx]
                
                if r_size == 1:
                    gen_schoolbook(1, m, c_size, r_start, 0, c_start)
                else:
                    # r_size == 2, split middle into 2+2
                    ms = middle_splits_per_col_group[c_idx]
                    mid_offsets = [0]
                    for mm in ms:
                        mid_offsets.append(mid_offsets[-1] + mm)
                    
                    for m_idx in range(len(ms)):
                        m_start = mid_offsets[m_idx]
                        m_size = ms[m_idx]
                        
                        if m_size == 2:
                            if c_size == 2:
                                gen_strassen_2x2x2(r_start, m_start, c_start)
                            elif c_size == 3:
                                gen_2x2x2x3(r_start, m_start, c_start)
                            elif c_size == 4:
                                gen_2x2x2x4(r_start, m_start, c_start)
                            elif c_size == 7:
                                gen_2x2x2x7(r_start, m_start, c_start)
                            else:
                                gen_schoolbook(2, 2, c_size, r_start, m_start, c_start)
                        else:
                            gen_schoolbook(2, m_size, c_size, r_start, m_start, c_start)

        if len(U_vecs) >= best_rank:
            return None

        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        
        # Verify
        ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
        jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
        kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
        
        U_idx = ii * m + jj
        V_idx = jj * p + kk
        W_idx = kk * n + ii
        
        U_sel = U[U_idx]
        V_sel = V[V_idx]
        W_sel = W[W_idx]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if not np.allclose(vals, 1.0, atol=1e-9):
            return None
        
        all_half_int = np.allclose(U * 2, np.round(U * 2)) and \
                       np.allclose(V * 2, np.round(V * 2)) and \
                       np.allclose(W * 2, np.round(W * 2))
        
        if not all_half_int:
            return None
        
        rank = U.shape[1]
        if rank < best_rank:
            return {"U": (np.round(U * 2) / 2).tolist(),
                    "V": (np.round(V * 2) / 2).tolist(),
                    "W": (np.round(W * 2) / 2).tolist(),
                    "rank": rank}
        return None

    # Strategy 1: rows 2+1, cols 3+4, middle 2+2 for each
    result = try_decomposition([2, 1], [3, 4], [[2, 2], [2, 2]])
    if result:
        best_result = result
        best_rank = result["rank"]

    # Strategy 2: rows 2+1, cols 4+3, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([2, 1], [4, 3], [[2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 3: rows 1+2, cols 3+4, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([1, 2], [3, 4], [[2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 4: rows 2+1, cols 2+2+3, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([2, 1], [2, 2, 3], [[2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 5: rows 2+1, cols 2+3+2, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([2, 1], [2, 3, 2], [[2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 6: rows 2+1, cols 2+2+2+1, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([2, 1], [2, 2, 2, 1], [[2, 2], [2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 7: rows 2+1, cols 1+2+2+2, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([2, 1], [1, 2, 2, 2], [[2, 2], [2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 8: rows 2+1, cols 2+2+1+2, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([2, 1], [2, 2, 1, 2], [[2, 2], [2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 9: rows 1+2, cols 2+2+3, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([1, 2], [2, 2, 3], [[2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    # Strategy 10: rows 1+2, cols 2+2+2+1, middle 2+2 for each
    if time.time() < deadline:
        result = try_decomposition([1, 2], [2, 2, 2, 1], [[2, 2], [2, 2], [2, 2], [2, 2]])
        if result and result["rank"] < best_rank:
            best_result = result
            best_rank = result["rank"]

    if best_result:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

    # Fallback: schoolbook rank 84
    U_vecs_fb = []
    V_vecs_fb = []
    W_vecs_fb = []
    
    def add_rank1_term_fb(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs_fb.append(u)
        V_vecs_fb.append(v)
        W_vecs_fb.append(w)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_rank1_term_fb([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
    
    U_fb = np.array(U_vecs_fb).T
    V_fb = np.array(V_vecs_fb).T
    W_fb = np.array(W_vecs_fb).T
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
