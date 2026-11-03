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

    def strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """Add Strassen 2x2x2 decomposition terms."""
        # M1 = (A11+A22)(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])
        # M2 = (A21+A22)B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])
        # M3 = A11(B21-B11)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])
        # M4 = A22(B12-B21)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])
        # M5 = (A11+A12)B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])
        # M6 = (A21-A11)(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])
        # M7 = (A12-A22)(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def block_2x2x1(i0, i1, j0, j1, k):
        """Add 2x2x1 decomposition terms (rank 4)."""
        for ii in [i0, i1]:
            for jj in [j0, j1]:
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k], [1.0],
                               [k*n+ii], [1.0])

    def block_1xmxp(i, j_off, p_l):
        """Add 1xmxp decomposition terms (schoolbook, rank m*p_l)."""
        for jj in range(j_off, j_off + m):
            for kk in range(p_l):
                add_rank1_term([i*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i], [1.0])

    def block_2x2x7(i_off, j_off, k_off):
        """2x2x7: three 2x2x2 Strassen blocks + one 2x2x1 block."""
        # Three 2x2x2 blocks covering k_off..k_off+5
        for kb in range(3):
            k0 = k_off + kb * 2
            k1 = k0 + 1
            strassen_2x2x2(i_off, i_off+1, j_off, j_off+1, k0, k1)
        # One 2x2x1 block for k_off+6
        block_2x2x1(i_off, i_off+1, j_off, j_off+1, k_off+6)

    def block_2x4x7(i_off, j_off, k_off):
        """2x4x7: two 2x2x7 blocks."""
        block_2x2x7(i_off, j_off, k_off)
        block_2x2x7(i_off, j_off+2, k_off)

    def block_1x4x7(i, j_off, k_off):
        """1x4x7: schoolbook, rank 28."""
        block_1xmxp(i, j_off, 7)

    # Main decomposition: 3x4x7 = 2x4x7 + 1x4x7
    # 2x4x7 uses 2*25 = 50 terms
    # 1x4x7 uses 28 terms
    # Total: 78 terms
    block_2x4x7(0, 0, 0)
    block_1x4x7(2, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Verify correctness
    rank = len(U_vecs)
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = np.sum(U[ci*m + cj] * V[cj*p + ck] * W[ck*n + ci])
                if abs(prod_sum - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook
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
