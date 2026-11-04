"""Matrix multiplication 3x4x6, block-Strassen hybrid exact decomposition (rank 54)."""
n, m, p = 3, 4, 6

# A index (r, x) -> U index
def uidx(r, x):
    return (r * m + x, 'U')
# B index (x, c) -> V index
def vid(x, c):
    return (x * p + c, 'V')
# C/W index (r, c) -> W index, trace convention W[c*n + r]
def wid(r, c):
    return (c * n + r, 'W')

def _mkcols():
    cols = []

    def strassen2x2x2x2(r0, x0, c0):
        def U(r, x):
            return (r0 + r, x0 + x)
        def V(x, c):
            return (x0 + x, c0 + c)
        def Wc(r, c, s):
            return (r0 + r, c0 + c, s)
        def add(Apos, Bpos, w):
            t = [(uidx(*a)[0], 'U', +1) for a in Apos]
            t += [(vid(*b)[0], 'V', +1) for b in Bpos]
            r, c, s = w
            t.append((wid(r, c)[0], 'W', s))
            cols.append(t)
        a = lambda r, x: [(r, x)]
        b = lambda x, c: [(x, c)]
        add(a(0, 0), b(0, 0), (0, 0, 1))                    # M1
        add(a(1, 0), b(0, 1) + b(1, 1), (0, 1, 1))          # M2 -> C01
        add(a(1, 0), b(0, 1) + b(1, 1), (1, 0, 1))          # M2 -> C10
        add(a(0, 0) + a(0, 1), b(1, 0), (0, 1, 1))          # M3 -> C01
        add(a(0, 0) + a(0, 1), b(1, 0), (1, 0, -1))         # M3 -> C10
        add(a(1, 1), b(0, 0) + b(1, 1), (0, 0, 1))          # M4 -> C00
        add(a(1, 1), b(0, 0) + b(1, 1), (1, 0, 1))          # M4 -> C10
        add(a(1, 0) + a(1, 1), b(0, 1), (0, 0, -1))         # M5 -> C00
        add(a(1, 0) + a(1, 1), b(0, 1), (1, 1, 1))          # M5 -> C11
        add(a(0, 0) + a(1, 0), b(1, 0) + b(1, 1), (0, 0, 1))# M6 -> C00
        add(a(0, 1) + a(1, 1), b(0, 0) + b(1, 0), (1, 1, 1))# M7 -> C11

    def direct1row(r0, x0, cx0, nx, nc):
        for x in range(nx):
            for c in range(nc):
                cols.append([(uidx(r0, x0 + x)[0], 'U', 1),
                             (vid(x0 + x, cx0 + c)[0], 'V', 1),
                             (wid(r0, cx0 + c)[0], 'W', 1)])

    # top 2 rows of A, for each 2-col group of B, split 4 x-rows into two 2-groups (Strassen each)
    for cg in (0, 2, 4):
        for xg in (0, 2):
            strassen2x2x2x2(r0=0, x0=xg, c0=cg)
    # bottom row r=2: direct 1x4 for each 2-col group
    for cg in (0, 2, 4):
        direct1row(r0=2, x0=0, cx0=cg, nx=4, nc=2)
    return cols

# EVOLVE-BLOCK-START
def construct():
    cols = _mkcols()
    rank = len(cols)
    nU, nV, nW = n * m, m * p, p * n
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]
    for ci, col in enumerate(cols):
        for (idx, tag, sgn) in col:
            if tag == 'U':
                U[idx][ci] += sgn
            elif tag == 'V':
                V[idx][ci] += sgn
            else:
                W[idx][ci] += sgn
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
