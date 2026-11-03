import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 3, 4, 7

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

    def strassen_2x2x2(i_off, j_off, k_off):
        i0 = i_off; i1 = i_off + 1
        j0 = j_off; j1 = j_off + 1
        k0 = k_off; k1 = k_off + 1

        add_rank1_term(
            [i0*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term(
            [i1*m+j0, i1*m+j1], [1, 1],
            [j0*p+k0], [1],
            [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term(
            [i0*m+j0], [1],
            [j0*p+k1, j1*p+k1], [1, -1],
            [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term(
            [i1*m+j1], [1],
            [j1*p+k0, j0*p+k0], [1, -1],
            [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term(
            [i0*m+j0, i0*m+j1], [1, 1],
            [j1*p+k1], [1],
            [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term(
            [i1*m+j0, i0*m+j0], [1, -1],
            [j0*p+k0, j0*p+k1], [1, 1],
            [k1*n+i1], [1])
        add_rank1_term(
            [i0*m+j1, i1*m+j1], [1, -1],
            [j1*p+k0, j1*p+k1], [1, 1],
            [k0*n+i0], [1])

    def schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 or m_l == 1 or p_l == 1:
            schoolbook_block(n_l, m_l, p_l, i_off, j_off, k_off)
            return
        if n_l == 2 and m_l == 2 and p_l == 2:
            strassen_2x2x2(i_off, j_off, k_off)
            return

        # Recursive splitting
        if n_l >= 2:
            n1 = n_l // 2
            gen_terms(n1, m_l, p_l, i_off, j_off, k_off)
            gen_terms(n_l - n1, m_l, p_l, i_off + n1, j_off, k_off)
            return
        if m_l >= 2:
            m1 = m_l // 2
            gen_terms(n_l, m1, p_l, i_off, j_off, k_off)
            gen_terms(n_l, m_l - m1, p_l, i_off, j_off + m1, k_off)
            return
        if p_l >= 2:
            p1 = p_l // 2
            gen_terms(n_l, m_l, p1, i_off, j_off, k_off)
            gen_terms(n_l, m_l, p_l - p1, i_off, j_off, k_off + p1)
            return

    gen_terms(n, m, p, 0, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
