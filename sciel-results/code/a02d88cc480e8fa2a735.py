import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    n, m, p = 3, 4, 6

    rank = 66
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    r = 0

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def add_rank1(u_entries, v_entries, w_entries):
        nonlocal r
        if r >= rank:
            return
        for (idx, val) in u_entries:
            U[idx, r] += val
        for (idx, val) in v_entries:
            V[idx, r] += val
        for (idx, val) in w_entries:
            W[idx, r] += val
        r += 1

    def schoolbook_1x4x2(i0, j0, k0):
        nonlocal r
        for jj in range(4):
            for kk in range(2):
                add_rank1(
                    [(idx_u(i0, j0 + jj), 1)],
                    [(idx_v(j0 + jj, k0 + kk), 1)],
                    [(idx_w(k0 + kk, i0), 1)]
                )

    def strassen_2x2x2(i0, j0, k0):
        nonlocal r
        def u(i, j):
            return idx_u(i0 + i, j0 + j)
        def v(j, k):
            return idx_v(j0 + j, k0 + k)
        def w(k, i):
            return idx_w(k0 + k, i0 + i)

        # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
        add_rank1(
            [(u(0,0), 1), (u(1,1), 1)],
            [(v(0,0), 1), (v(1,1), 1)],
            [(w(0,0), 1), (w(1,1), 1)]
        )
        # m2 = (a10+a11)(b00) -> c10 += m2, c11 -= m2
        add_rank1(
            [(u(1,0), 1), (u(1,1), 1)],
            [(v(0,0), 1)],
            [(w(0,1), 1), (w(1,1), -1)]
        )
        # m3 = (a00)(b01-b11) -> c01 += m3, c11 += m3
        add_rank1(
            [(u(0,0), 1)],
            [(v(0,1), 1), (v(1,1), -1)],
            [(w(1,0), 1), (w(1,1), 1)]
        )
        # m4 = (a11)(b10-b00) -> c00 += m4, c10 += m4
        add_rank1(
            [(u(1,1), 1)],
            [(v(1,0), 1), (v(0,0), -1)],
            [(w(0,0), 1), (w(0,1), 1)]
        )
        # m5 = (a00+a01)(b11) -> c00 -= m5, c01 += m5
        add_rank1(
            [(u(0,0), 1), (u(0,1), 1)],
            [(v(1,1), 1)],
            [(w(0,0), -1), (w(1,0), 1)]
        )
        # m6 = (a10-a00)(b00+b01) -> c11 += m6
        add_rank1(
            [(u(1,0), 1), (u(0,0), -1)],
            [(v(0,0), 1), (v(0,1), 1)],
            [(w(1,1), 1)]
        )
        # m7 = (a01-a11)(b10+b11) -> c00 += m7
        add_rank1(
            [(u(0,1), 1), (u(1,1), -1)],
            [(v(1,0), 1), (v(1,1), 1)],
            [(w(0,0), 1)]
        )

    # M(3,4,6) = 3 blocks of M(3,4,2) with k split 2+2+2
    for kb in range(3):
        if time.time() > deadline:
            break
        k0 = kb * 2
        # M(1,4,2) for row 0
        schoolbook_1x4x2(0, 0, k0)
        # M(2,4,2) for rows 1,2 = 2 * M(2,2,2) with j split 2+2
        strassen_2x2x2(1, 0, k0)
        strassen_2x2x2(1, 2, k0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
