import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = DIMENSIONS

    U_list = []
    V_list = []
    W_list = []

    def emit(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def emit_term(ii, kk, jj, cu=1.0, cv=1.0, cw=1.0):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ii * m + kk] = cu
        v[kk * p + jj] = cv
        w[jj * n + ii] = cw
        emit(u, v, w)

    # Strassen decomposition for the 4x4x4 core (indices 0..3 in all dims)
    # 7 Strassen products, each expanded into 4 schoolbook 2x2x2 terms
    sp = [
        ({'a': [(0,1),(3,1)], 'b': [(0,1),(3,1)]}, {0:1, 3:1}),
        ({'a': [(2,1),(3,1)], 'b': [(0,1)]},     {2:1, 3:-1}),
        ({'a': [(0,1)], 'b': [(1,1),(3,-1)]},    {1:1, 3:1}),
        ({'a': [(3,1)], 'b': [(2,1),(0,-1)]},    {0:1, 2:1}),
        ({'a': [(0,1),(1,1)], 'b': [(3,1)]},     {0:-1, 1:1}),
        ({'a': [(2,1),(0,-1)], 'b': [(0,1),(1,1)]}, {3:1}),
        ({'a': [(1,1),(3,-1)], 'b': [(2,1),(3,1)]}, {0:1}),
    ]

    def ag(bl, li, lk):
        if bl == 0: return (li, lk)
        if bl == 1: return (li, lk+2)
        if bl == 2: return (li+2, lk)
        return (li+2, lk+2)

    def bg(bl, lk, lj):
        if bl == 0: return (lk, lj)
        if bl == 1: return (lk, lj+2)
        if bl == 2: return (lk+2, lj)
        return (lk+2, lj+2)

    def cg(bl, li, lj):
        if bl == 0: return (li, lj)
        if bl == 1: return (li, lj+2)
        if bl == 2: return (li+2, lj)
        return (li+2, lj+2)

    for prod, ccontrib in sp:
        aform = prod['a']
        bform = prod['b']
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    uv = [0.0] * (n * m)
                    vv = [0.0] * (m * p)
                    wv = [0.0] * (p * n)
                    for abl, a_sign in aform:
                        gi, gk = ag(abl, li, lk)
                        uv[gi * m + gk] = a_sign
                    for bbl, b_sign in bform:
                        gk, gj = bg(bbl, lk, lj)
                        vv[gk * p + gj] = b_sign
                    for cbl, c_sign in ccontrib.items():
                        gi, gj = cg(cbl, li, lj)
                        wv[gj * n + gi] = c_sign
                    emit(uv, vv, wv)

    # Tail terms: j=4 (last column of C) and k=4 (last row of A)
    for ii in range(4):
        for kk in range(4):
            emit_term(ii, kk, 4)
    for ii in range(4):
        for jj in range(5):
            emit_term(ii, 4, jj)

    if not U_list:
        emit_term(0, 0, 0)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
