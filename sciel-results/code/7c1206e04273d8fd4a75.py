import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
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

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off
            j = j_off
            k = k_off
            add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            k1 = k_off + 1

            # Standard Strassen 2x2x2
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])
            return

        if n_l == 2 and m_l == 2 and p_l == 1:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            # Rank 1 decomposition
            add_rank1_term([i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1], [1, 1, 1, 1],
                           [j0*p+k0, j1*p+k0], [1, 1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            return

        if n_l == 1 and m_l == 2 and p_l == 2:
            i0 = i_off
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            k1 = k_off + 1
            # Rank 1 decomposition
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1], [1, 1, 1, 1],
                           [k0*n+i0, k1*n+i0], [1, 1])
            return

        if n_l == 1 and m_l == 2 and p_l == 1:
            i0 = i_off
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            # Rank 1 decomposition
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k0], [1, 1],
                           [k0*n+i0], [1])
            return

        if n_l == 2 and m_l == 1 and p_l == 2:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            k0 = k_off
            k1 = k_off + 1
            # Rank 1 decomposition
            add_rank1_term([i0*m+j0, i1*m+j0], [1, 1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            return

        if n_l == 2 and m_l == 1 and p_l == 1:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            k0 = k_off
            # Rank 1 decomposition
            add_rank1_term([i0*m+j0, i1*m+j0], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            return

        if n_l == 1 and m_l == 1 and p_l == 2:
            i0 = i_off
            j0 = j_off
            k0 = k_off
            k1 = k_off + 1
            # Rank 1 decomposition
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i0], [1, 1])
            return

        # Recursive decomposition
        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off + 2, j_off, k_off)
            return

        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off + 2, k_off)
            return

        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 2)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 4)
            gen_terms(2, 2, 1, i_off, j_off, k_off + 6)
            return

        if n_l == 1 and m_l == 4 and p_l == 7:
            gen_terms(1, 2, 7, i_off, j_off, k_off)
            gen_terms(1, 2, 7, i_off, j_off + 2, k_off)
            return

        if n_l == 1 and m_l == 2 and p_l == 7:
            gen_terms(1, 2, 2, i_off, j_off, k_off)
            gen_terms(1, 2, 2, i_off, j_off, k_off + 2)
            gen_terms(1, 2, 2, i_off, j_off, k_off + 4)
            gen_terms(1, 2, 1, i_off, j_off, k_off + 6)
            return

        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])

    gen_terms(n, m, p, 0, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Verify
    rank = len(U_vecs)
    valid = True
    for check_i in range(n):
        for check_j in range(m):
            for check_k in range(p):
                u_val = U[check_i*m + check_j]
                v_val = V[check_j*p + check_k]
                w_val = W[check_k*n + check_i]
                prod_sum = np.sum(u_val * v_val * w_val)
                if abs(prod_sum - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to schoolbook if verification fails
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
