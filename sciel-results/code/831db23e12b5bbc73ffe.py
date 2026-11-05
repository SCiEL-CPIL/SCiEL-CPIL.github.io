import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0  # explicit deadline guard with safety margin

    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24

    def time_check():
        return time.time() > _deadline

    terms = []  # list of (u_sp, v_sp, w_sp) where each *_sp is a list of (index, value)

    def add_strassen_block(i0, j0, k0):
        if time_check():
            return
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        blocks = [
            ([(a00, 1.0), (a11, 1.0)], [(b00, 1.0), (b11, 1.0)], [(c00, 1.0), (c11, 1.0)]),
            ([(a10, 1.0), (a11, 1.0)], [(b00, 1.0)], [(c10, 1.0), (c11, -1.0)]),
            ([(a00, 1.0)], [(b01, 1.0), (b11, -1.0)], [(c01, 1.0), (c11, 1.0)]),
            ([(a11, 1.0)], [(b10, 1.0), (b00, -1.0)], [(c00, 1.0), (c10, 1.0)]),
            ([(a00, 1.0), (a01, 1.0)], [(b11, 1.0)], [(c00, -1.0), (c01, 1.0)]),
            ([(a10, 1.0), (a00, -1.0)], [(b00, 1.0), (b01, 1.0)], [(c11, 1.0)]),
            ([(a01, 1.0), (a11, -1.0)], [(b10, 1.0), (b11, 1.0)], [(c00, 1.0)]),
        ]
        for u, v, w in blocks:
            terms.append((u, v, w))

    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        if time_check():
            return
        for j in range(m):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                terms.append(([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)]))

    # A1 (rows 0,1) x B1 (cols 0-3): 4 Strassen 2x2x2 blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    # A1 (rows 0,1) x B2 (cols 4-7): 4 Strassen 2x2x2 blocks
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    add_strassen_block(0, 2, 4)
    add_strassen_block(0, 2, 6)
    # A2 (row 2) x B1 (cols 0-3): schoolbook 1x4x4
    add_schoolbook_1x4_4x4(2, 0, 0)
    # A2 (row 2) x B2 (cols 4-7): schoolbook 1x4x4
    add_schoolbook_1x4_4x4(2, 0, 4)

    rank = len(terms)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u:
            U[idx, r] = val
        for idx, val in v:
            V[idx, r] = val
        for idx, val in w:
            W[idx, r] = val

    # Verification with fallback
    if not time_check():
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        T_true = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0
        if np.max(np.abs(T_decomp - T_true)) > 1e-9:
            R = n * m * p
            U = np.zeros((nm, R)); V = np.zeros((mp, R)); W = np.zeros((pn, R))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
                        r += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
