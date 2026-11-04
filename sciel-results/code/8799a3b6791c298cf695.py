"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Uses a recursive block decomposition with Strassen for 2x2x2 blocks,
and tries multiple partition strategies to find the lowest-rank valid decomposition.
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

    def build_decomposition(n_split, col_splits, middle_splits_per_col_group):
        """Build a decomposition for a given partition strategy."""
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

        def gen_block(n_l, m_l, p_l, i_off, j_off, k_off):
            """Recursively handle a block with optimal sub-decomposition."""
            if n_l == 0 or m_l == 0 or p_l == 0:
                return
            
            if n_l == 1:
                gen_schoolbook(1, m_l, p_l, i_off, j_off, k_off)
                return
            
            if m_l == 1:
                gen_schoolbook(n_l, 1, p_l, i_off, j_off, k_off)
                return
            
            if p_l == 1:
                gen_schoolbook(n_l, m_l, 1, i_off, j_off, k_off)
                return
            
            if n_l == 2 and m_l == 2 and p_l == 2:
                gen_strassen_2x2x2(i_off, j_off, k_off)
                return
            
            if n_l == 2 and m_l == 2:
                # Split p into 2s and maybe a 1
                k = 0
                while k + 2 <= p_l:
                    gen_strassen_2x2x2(i_off, j_off, k_off + k)
                    k += 2
                if k < p_l:
                    gen_schoolbook(2, 2, 1, i_off, j_off, k_off + k)
                return
            
            if n_l == 2 and m_l == 3:
                # Split m into 2+1
                gen_block(2, 2, p_l, i_off, j_off, k_off)
                gen_block(2, 1, p_l, i_off, j_off+2, k_off)
                return
            
            if n_l == 2 and m_l == 4:
                # Split m into 2+2
                gen_block(2, 2, p_l, i_off, j_off, k_off)
                gen_block(2, 2, p_l, i_off, j_off+2, k_off)
                return
            
            if n_l == 2 and m_l == 5:
                # Split m into 2+2+1
                gen_block(2, 2, p_l, i_off, j_off, k_off)
                gen_block(2, 2, p_l, i_off, j_off+2, k_off)
                gen_block(2, 1, p_l, i_off, j_off+4, k_off)
                return
            
            if n_l == 2 and m_l == 6:
                # Split m into 2+2+2
                gen_block(2, 2, p_l, i_off, j_off, k_off)
                gen_block(2, 2, p_l, i_off, j_off+2, k_off)
                gen_block(2, 2, p_l, i_off, j_off+4, k_off)
                return
            
            if n_l == 2 and m_l == 7:
                # Split m into 2+2+2+1
                gen_block(2, 2, p_l, i_off, j_off, k_off)
                gen_block(2, 2, p_l, i_off, j_off+2, k_off)
                gen_block(2, 2, p_l, i_off, j_off+4, k_off)
                gen_block(2, 1, p_l, i_off, j_off+6, k_off)
                return
            
            if n_l == 3:
                # Split n into 2+1
                gen_block(2, m_l, p_l, i_off, j_off, k_off)
                gen_block(1, m_l, p_l, i_off+2, j_off, k_off)
                return
            
            # Fallback: schoolbook
            gen_schoolbook(n_l, m_l, p_l, i_off, j_off, k_off)

        # Build the decomposition using the recursive block approach
        # But we also want to try the explicit partition strategies
        # For now, use the recursive approach which should be equivalent
        
        # Actually, let's use the explicit partition approach for more control
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
                    # r_size == 2, split middle into 2s
                    ms = middle_splits_per_col_group[c_idx]
                    mid_offsets = [0]
                    for mm in ms:
                        mid_offsets.append(mid_offsets[-1] + mm)
                    
                    for m_idx in range(len(ms)):
                        m_start = mid_offsets[m_idx]
                        m_size = ms[m_idx]
                        
                        if m_size == 2:
                            if c_size == 1:
                                gen_schoolbook(2, 2, 1, r_start, m_start, c_start)
                            elif c_size == 2:
                                gen_strassen_2x2x2(r_start, m_start, c_start)
                            else:
                                # Split c_size into 2s and maybe a 1
                                k = 0
                                while k + 2 <= c_size:
                                    gen_strassen_2x2x2(r_start, m_start, c_start + k)
                                    k += 2
                                if k < c_size:
                                    gen_schoolbook(2, 2, 1, r_start, m_start, c_start + k)
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

    # Try various partition strategies
    strategies = [
        ([2, 1], [3, 4], [[2, 2], [2, 2]]),
        ([2, 1], [4, 3], [[2, 2], [2, 2]]),
        ([1, 2], [3, 4], [[2, 2], [2, 2]]),
        ([2, 1], [2, 2, 3], [[2, 2], [2, 2], [2, 2]]),
        ([2, 1], [2, 3, 2], [[2, 2], [2, 2], [2, 2]]),
        ([2, 1], [2, 5], [[2, 2], [2, 2]]),
        ([2, 1], [5, 2], [[2, 2], [2, 2]]),
        ([2, 1], [1, 6], [[2, 2], [2, 2]]),
        ([2, 1], [6, 1], [[2, 2], [2, 2]]),
        ([1, 2], [2, 5], [[2, 2], [2, 2]]),
        ([2, 1], [3, 2, 2], [[2, 2], [2, 2], [2, 2]]),
        ([2, 1], [2, 2, 2, 1], [[2, 2], [2, 2], [2, 2], [2, 2]]),
        ([2, 1], [1, 2, 2, 2], [[2, 2], [2, 2], [2, 2], [2, 2]]),
        ([2, 1], [2, 1, 2, 2], [[2, 2], [2, 2], [2, 2], [2, 2]]),
        ([2, 1], [2, 2, 1, 2], [[2, 2], [2, 2], [2, 2], [2, 2]]),
    ]
    
    for n_split, col_splits, mid_splits in strategies:
        if time.time() > deadline:
            break
        result = build_decomposition(n_split, col_splits, mid_splits)
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
