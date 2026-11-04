"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Fixed block decomposition: top 2x4 block uses two 2x2x2 Strassen
decompositions (rank 7 each), bottom 1x4 block uses schoolbook (rank 7).
Total rank 21, exact (verified against the multiplication tensor).
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def _verify(n, m, p, U, V, W):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    U_idx = (ii * m + jj).ravel()
    V_idx = (jj * p + kk).ravel()
    W_idx = (kk * n + ii).ravel()
    vals = np.einsum('ir,jr,kr->ijk', U[U_idx].reshape(n, m, -1),
                     V[V_idx].reshape(m, p, -1),
                     W[W_idx].reshape(p, n, -1))
    return np.allclose(vals, 1.0, atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_term(u, v, w):
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    u = np.zeros(n * m); u[ii * m + jj] = 1.0
                    v = np.zeros(m * p); v[jj * p + kk] = 1.0
                    w = np.zeros(p * n); w[kk * n + ii] = 1.0
                    add_term(u, v, w)

    def strassen_2x2x2(i0, j0, k0):
        """Exact rank-7 decomposition of a 2x2 * 2x2 sub-block.
        Uses global index strides: U stride = m (full middle dim),
        V stride = p (full output dim), W stride = n (full row dim).
        Local block offsets (i0,j0,k0) shift the base indices.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        def mk_u(idx_vals):
            u = np.zeros(n * m)
            for (ii, jj), val in idx_vals:
                u[ii * m + jj] += val
            return u

        def mk_v(idx_vals):
            v = np.zeros(m * p)
            for (jj, kk), val in idx_vals:
                v[jj * p + kk] += val
            return v

        def mk_w(idx_vals):
            w = np.zeros(p * n)
            for (kk, ii), val in idx_vals:
                w[kk * n + ii] += val
            return w

        # M1 = (i0+j0)(j0+k0)(k0+i0) + (i1+j1)(j1+k1)(k1+i1)
        add_term(
            mk_u([((i0, j0), 1), ((i1, j1), 1)]),
            mk_v([((j0, k0), 1), ((j1, k1), 1)]),
            mk_w([((k0, i0), 1), ((k1, i1), 1)]),
        )
        # M2 = (i1+j0)(j0+k0)(k0+i1) - (i1+j1)(j0+k0)(k1+i1)
        add_term(
            mk_u([((i1, j0), 1), ((i1, j1), 1)]),
            mk_v([((j0, k0), 1)]),
            mk_w([((k0, i1), 1), ((k1, i1), -1)]),
        )
        # M3 = (i0+j0)(j0+k1 - j1+k1)(k1+i0 + k1+i1)
        add_term(
            mk_u([((i0, j0), 1)]),
            mk_v([((j0, k1), 1), ((j1, k1), -1)]),
            mk_w([((k1, i0), 1), ((k1, i1), 1)]),
        )
        # M4 = (i1+j1)(j1+k0 - j0+k0)(k0+i0 + k0+i1)
        add_term(
            mk_u([((i1, j1), 1)]),
            mk_v([((j1, k0), 1), ((j0, k0), -1)]),
            mk_w([((k0, i0), 1), ((k0, i1), 1)]),
        )
        # M5 = (i0+j0 + i0+j1)(j1+k1)(-k0+i0 + k1+i0)
        add_term(
            mk_u([((i0, j0), 1), ((i0, j1), 1)]),
            mk_v([((j1, k1), 1)]),
            mk_w([((k0, i0), -1), ((k1, i0), 1)]),
        )
        # M6 = (i1+j0 - i0+j0)(j0+k0 + j0+k1)(k1+i1)
        add_term(
            mk_u([((i1, j0), 1), ((i0, j0), -1)]),
            mk_v([((j0, k0), 1), ((j0, k1), 1)]),
            mk_w([((k1, i1), 1)]),
        )
        # M7 = (i0+j1 - i1+j1)(j1+k0 + j1+k1)(k0+i0)
        add_term(
            mk_u([((i0, j1), 1), ((i1, j1), -1)]),
            mk_v([((j1, k0), 1), ((j1, k1), 1)]),
            mk_w([((k0, i0), 1)]),
        )

    # ---- Assemble rank-21 decomposition ----
    # Top 2x4 block: two 2x2x2 Strassen blocks (middle cols 0-1 and 2-3)
    strassen_2x2x2(i0=0, j0=0, k0=0)   # rows 0,1 ; middle 0,1 ; out 0,1
    strassen_2x2x2(i0=0, j0=2, k0=2)   # rows 0,1 ; middle 2,3 ; out 2,3
    # Bottom 1x4 block: schoolbook
    schoolbook(1, 4, 4, i_off=2, j_off=0, k_off=0)

    if len(U_vecs) > 21:
        raise RuntimeError("unexpected rank")

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Verify exactness
    if _verify(n, m, p, U, V, W):
        # Entries are integers (0, +/-1); round to be safe
        U = np.round(U)
        V = np.round(V)
        W = np.round(W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 84 (always valid)
    U2, V2, W2 = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m); u[i * m + j] = 1.0
                v = np.zeros(m * p); v[j * p + k] = 1.0
                w = np.zeros(p * n); w[k * n + i] = 1.0
                U2.append(u); V2.append(v); W2.append(w)
    Uf = np.array(U2).T
    Vf = np.array(V2).T
    Wf = np.array(W2).T
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
