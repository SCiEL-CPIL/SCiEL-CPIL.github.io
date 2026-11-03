import numpy as np
import time

_START = time.time()
DEADLINE = 150.0
MARGIN = 5.0

DIMENSIONS = [3, 3, 8]


def _time_left():
    return DEADLINE - MARGIN - (time.time() - _START)


def _verify(U, V, W, n, m, p):
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                if abs(s - 1.0) > 1e-9:
                    ok = False
    return ok


def _strassen_2x2_2():
    # U: [1,0,0,1] (A00+A11), [0,0,1,1] (A10+A11), [1,0,0,0] (A00), 
    #    [0,0,0,1] (A11), [1,1,0,0] (A00+A01), [-1,0,1,0] (A10-A00), [0,1,0,-1] (A01-A11)
    u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # V: [1,0,0,1] (B00+B11), [1,0,0,0] (B00), [0,1,0,-1] (B10-B11), 
    #    [0,0,1,-1] (B01-B00), [0,0,0,1] (B11), [1,1,0,0] (B00+B01), [0,0,1,1] (B10+B11)
    v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # W indices: 0=C00, 1=C10, 2=C01, 3=C11 (based on W[k*n+i], n=2, k=0,1; i=0,1)
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 + M2 - M3 - M6
    
    # M1: C00=1, C11=1
    # M2: C10=1
    # M3: C01=1
    # M4: C00=-1, C10=1
    # M5: C00=1, C01=1
    # M6: C11=-1
    # M7: C00=-1
    
    w = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, 0],   # M2
        [0, 0, 1, 0],   # M3
        [-1, 1, 0, 0],  # M4
        [1, 0, 1, 0],   # M5
        [0, 0, 0, -1],  # M6
        [-1, 0, 0, 0]   # M7
    ]
    return u, v, w


def _embed_strassen_into_3x3x8():
    n, m, p = DIMENSIONS
    blocks = [
        (0, 2, 0, 2, 0, 2),
        (0, 2, 0, 2, 2, 4),
        (0, 2, 0, 2, 4, 6),
        (0, 2, 0, 2, 6, 8),
    ]
    rank = 0
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    covered.add((i, j, k))
    schoolbook_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_count += 1
    rank = 4 * 7 + schoolbook_count

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    su, sv, sw = _strassen_2x2_2()
    col = 0
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    local = li * 2 + lj
                    uval = su[t][local]
                    if uval != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        U[gi * m + gj][col] += uval
            for lj in range(2):
                for lk in range(2):
                    local = lj * 2 + lk
                    vval = sv[t][local]
                    if vval != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] += vval
            for lk in range(2):
                for li in range(2):
                    local = li * 2 + lk
                    wval = sw[t][local]
                    if wval != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        W[gk * n + gi][col] += wval
            col += 1

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    try:
        if _time_left() > 1.0:
            res = _embed_strassen_into_3x3x8()
            if _verify(res["U"], res["V"], res["W"], n, m, p):
                return res
    except Exception:
        pass
    return _schoolbook()
