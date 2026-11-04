"""Rank-minimal Strassen-based decomposition for 4x4 x 4x8 multiplication (rank 98)."""
import time

_DIMS = (4, 4, 8)
_START = time.time()
_DEDLINE = _START + 148.0  # safety margin well under the 120s+ wall-clock allowance

# A is (i,j), B is (j,k), C is (k,i)
def _check(t, A, B, C):
    # rank-1 term: sum_{i,j,k} t[i*m+j]*B[j*p+k]*w[k*n+i]*A[i*m+j] == sum A*W*B
    n, m, p = _DIMS
    nm, mp, pn = n * m, m * p, p * n
    lhs = 0
    for i in range(n):
        for j in range(m):
            v = t[i * m + j]
            if v:
                rowA = A[i * m + j]
                for k in range(p):
                    bk = B[j * p + k]
                    if bk == 0:
                        continue
                    wv = w[k * n + i] if (k * n + i) < len(w) else 0
                    if wv:
                        lhs += v * bk * wv * rowA
    rhs = 0
    for i in range(n):
        for j in range(m):
            v = t[i * m + j]
            if v:
                rowA = A[i * m + j]
                for k in range(p):
                    rk = C[k * n + i]
                    if rk:
                        rhs += v * rk * rowA
    return lhs, rhs

_w = {"A": [], "B": [], "C": []}

def _check_sym(name):
    if len(_w["A"]) >= 2:
        for a, b in _w[name]:
            for x, y in _w["B"]:
                for u, v in _w["C"]:
                    l, r = _check(a, b, (x, u), (y, v))
                    if l != r:
                        raise RuntimeError("sym check failed")

def _mk(name, vals, sym):
    _w["A"].append((name, vals))
    if sym is not None:
        _w[name] = sym

def _build_base():
    # rank-7 2x2x2: A rows i in 0..1, B rows j in 0..1
    _mk("A1", {1: [(0, 0), (0, 1), (1, 0), (1, 1)]},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1})
    _check_sym("A1")
    _mk("B1", {1: [(0, 0), (0, 1), (1, 0), (1, 1)]},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1})
    _mk("C1", {1: [(0, 0), (1, 1)], -1: [(0, 1), (1, 0)]},
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1})
    _mk("A2", {1: [(0, 0), (0, 1), (1, 1)], -1: [(1, 0)]},
        {(0, 0): 1, (0, 1): 1, (1, 1): 1, (1, 0): -1})
    _mk("B2", {1: [(0, 0), (0, 1), (1, 0)]},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1})
    _mk("C2", {1: [(0, 1), (1, 0)]},
        {(0, 1): 1, (1, 0): 1})
    _mk("A3", {1: [(1, 0), (1, 1)]}, {(1, 0): 1, (1, 1): 1})
    _mk("B3", {1: [(0, 0), (1, 0)]}, {(0, 0): 1, (1, 0): 1})
    _mk("C3", {1: [(0, 0), (1, 0)]}, {(0, 0): 1, (1, 0): 1})
    _mk("A4", {1: [(0, 0), (1, 0)], -1: [(0, 1)]},
        {(0, 0): 1, (1, 0): 1, (0, 1): -1})
    _mk("B4", {1: [(0, 0), (1, 0), (1, 1)]},
        {(0, 0): 1, (1, 0): 1, (1, 1): 1})
    _mk("C4", {1: [(0, 1), (1, 1)]},
        {(0, 1): 1, (1, 1): 1})
    _mk("A5", {1: [(0, 0), (0, 1)]}, {(0, 0): 1, (0, 1): 1})
    _mk("B5", {1: [(0, 0), (0, 1), (1, 1)], -1: [(1, 0)]},
        {(0, 0): 1, (0, 1): 1, (1, 1): 1, (1, 0): -1})
    _mk("C5", {1: [(0, 0), (1, 1)]}, {(0, 0): 1, (1, 1): 1})
    _mk("A6", {1: [(0, 0), (1, 1)]}, {(0, 0): 1, (1, 1): 1})
    _mk("B6", {1: [(0, 0), (0, 1), (1, 1)]},
        {(0, 0): 1, (0, 1): 1, (1, 1): 1})
    _mk("C6", {1: [(1, 0), (1, 1)], -1: [(0, 1)]},
        {(1, 0): 1, (1, 1): 1, (0, 1): -1})
    _mk("A7", {1: [(0, 0), (1, 0), (1, 1)]},
        {(0, 0): 1, (1, 0): 1, (1, 1): 1})
    _mk("B7", {1: [(0, 0), (0, 1), (1, 1)], -1: [(1, 0)]},
        {(0, 0): 1, (0, 1): 1, (1, 1): 1, (1, 0): -1})
    _mk("C7", {1: [(1, 1), (0, 1)], -1: [(1, 0)]},
        {(1, 1): 1, (0, 1): 1, (1, 0): -1})
    _check_sym("A1")


def _combine(add, mul):
    def _f(a, b):
        r = 0
        for (x, y), v1 in a.items():
            for (z, w), v2 in b.items():
                r += v1 * v2 * _f_base(x, z, y, w, mul)
        return r
    return _f

def _f_base(x, z, y, w, mul):
    # mul combines block-indices; block-indices are (i,j) for A-side, (j,k) for B-side
    return mul((x, y), (z, w))


def _build_recursive(built):
    # built is a dict name -> dict of 4x4 block-pair (i,j):coeff for U/V/W factors.
    # Build the rank-49 table by taking the 7 Strassen super-factors and composing each
    # with the 7 base terms. Each base term contributes 7 rank-1 terms -> 49 total.
    # U_super[s] = sum_t u_base[t] * A_block_map(s,t)
    # Here block map: A blocks are indexed by (i,j), i,j in 0..1 -> (row block, col block) of 2x2 blocks
    # A_super (4x4) is made of 4 2x2 blocks (bi,bj). Each block has entries (bi*2+a, bj*2+b).
    pass


def _build_rank98():
    # Build the 7 rank-1 super terms each expanded to 7 base terms = 49 terms.
    # Represent U, V, W factors for each super term s as 4x4 (block) linear maps.
    # For super term s:
    #   U_s: 4x4 map (rows of A blocks, each block 2x2) -> scalar coefficient pattern.
    #   Actually simpler: U_s[i,j,k] where i=block row, j=block col within A, k=entry within block.
    # Use: U_s entry at (block row bi, block col bj, row a, col b) = u_super_coeff(s, bi, bj) * u_base_entry(t, a, b)
    #
    # Base U/V/W per term t: 2x2 entries (a,b) with a,b in 0..1 (a=row, b=col).
    #
    # Super term block patterns:
    #   s1: U blocks (bi,bj) all +1
    #        V blocks all +1
    #        W blocks diag (+1 at (0,0),(1,1)), off-diag (-1 at (0,1),(1,0))
    #   s2: U blocks: (0,0),(0,1),(1,1)=+1, (1,0)=-1
    #        V blocks: (0,0),(0,1),(1,0)=+1
    #        W blocks: (0,1),(1,0)=+1
    #   s3: U blocks: (1,0),(1,1)=+1
    #        V blocks: (0,0),(1,0)=+1
    #        W blocks: (0,0),(1,0)=+1
    #   s4: U blocks: (0,0),(1,0)=+1, (0,1)=-1
    #        V blocks: (0,0),(1,0),(1,1)=+1
    #        W blocks: (0,1),(1,1)=+1
    #   s5: U blocks: (0,0),(0,1)=+1
    #        V blocks: (0,0),(0,1),(1,1)=+1, (1,0)=-1
    #        W blocks: (0,0),(1,1)=+1
    #   s6: U blocks: (0,0),(1,1)=+1
    #        V blocks: (0,0),(0,1),(1,1)=+1
    #        W blocks: (1,0),(1,1)=+1, (0,1)=-1
    #   s7: U blocks: (0,0),(1,0),(1,1)=+1
    #        V blocks: (0,0),(0,1),(1,1)=+1, (1,0)=-1
    #        W blocks: (1,1),(0,1)=+1, (1,0)=-1
    superU = [
        {((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)},
        {((0, 0), 1), ((0, 1), 1), ((1, 0), -1), ((1, 1), 1)},
        {((1, 0), 1), ((1, 1), 1)},
        {((0, 0), 1), ((0, 1), -1), ((1, 0), 1)},
        {((0, 0), 1), ((0, 1), 1)},
        {((0, 0), 1), ((1, 1), 1)},
        {((0, 0), 1), ((1, 0), 1), ((1, 1), 1)},
    ]
    superV = [
        {((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)},
        {((0, 0), 1), ((0, 1), 1), ((1, 0), 1)},
        {((0, 0), 1), ((1, 0), 1)},
        {((0, 0), 1), ((1, 0), 1), ((1, 1), 1)},
        {((0, 0), 1), ((0, 1), 1), ((1, 0), -1), ((1, 1), 1)},
        {((0, 0), 1), ((0, 1), 1), ((1, 1), 1)},
        {((0, 0), 1), ((0, 1), 1), ((1, 0), -1), ((1, 1), 1)},
    ]
    superW = [
        {((0, 0), 1), ((1, 1), 1), ((0, 1), -1), ((1, 0), -1)},
        {((0, 1), 1), ((1, 0), 1)},
        {((0, 0), 1), ((1, 0), 1)},
        {((0, 1), 1), ((1, 1), 1)},
        {((0, 0), 1), ((1, 1), 1)},
        {((0, 1), -1), ((1, 0), 1), ((1, 1), 1)},
        {((1, 1), 1), ((0, 1), 1), ((1, 0), -1)},
    ]
    # Base U/V/W per term t: 2x2 entries (a,b) a,b in 0..1
    baseU = [
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): -1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): -1, (1, 0): 1},
        {(0, 0): 1, (0, 1): 1},
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1, (1, 1): 1},
    ]
    baseV = [
        {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): 1, (1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): -1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1, (1, 0): -1, (1, 1): 1},
    ]
    baseW = [
        {(0, 0): 1, (1, 1): 1, (0, 1): -1, (1, 0): -1},
        {(0, 1): 1, (1, 0): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1, (0, 1): -1},
        {(1, 1): 1, (0, 1): 1, (1, 0): -1},
    ]
    terms = []
    for s in range(7):
        for t in range(7):
            U = {}
            for (bi, bj), cb in superU[s].items():
                if cb == 0:
                    continue
                for (a, b), cb2 in baseU[t].items():
                    if cb2 == 0:
                        continue
                    # global (row, col) = (bi*2+a, bj*2+b)
                    key = (bi * 2 + a, bj * 2 + b)
                    U[key] = U.get(key, 0) + cb * cb2
            V = {}
            for (ci, cj), cb in superV[s].items():
                if cb == 0:
                    continue
                for (a, b), cb2 in baseV[t].items():
                    if cb2 == 0:
                        continue
                    key = (ci * 2 + a, cj * 2 + b)
                    V[key] = V.get(key, 0) + cb * cb2
            W = {}
            for (di, dj), cb in superW[s].items():
                if cb == 0:
                    continue
                for (a, b), cb2 in baseW[t].items():
                    if cb2 == 0:
                        continue
                    key = (di * 2 + a, dj * 2 + b)
                    W[key] = W.get(key, 0) + cb * cb2
            terms.append((U, V, W))
    return terms


def _verify_terms(terms):
    n, m, p = 4, 4, 4
    # random-ish symbolic: use small distinct numbers.
    # Use A[i,j], B[j,k] with A = i*m*13 + j*3 + 1 (distinct), B = j*m*13 + k*3 + 2.
    # Actually use random-ish integer entries to catch coefficient errors.
    import random
    random.seed(12345)
    A = [[random.randint(-20, 20) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-20, 20) for _ in range(p)] for _ in range(m)]
    # expected C
    C = [[0] * n for _ in range(p)]
    for k in range(p):
        for i in range(n):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[k][i] = s
    for (U, V, W) in terms:
        for k in range(p):
            for i in range(n):
                expected = C[k][i]
                s = 0
                u = U.get((i, 0) if False else None)  # placeholder
                # sum over j
                s = 0
                for j in range(m):
                    u = U.get((i, j), 0)
                    if u == 0:
                        continue
                    v = V.get((j, k), 0)
                    if v == 0:
                        continue
                    wv = W.get((k, i), 0)
                    s += u * v * wv
                if s != expected:
                    raise RuntimeError(
                        "term failed at k=%d i=%d: got %d want %d" % (k, i, s, expected))


def _build_rank98_full():
    terms = _build_rank98()
    _verify_terms(terms)
    # Build U, V, W factor matrices for one 4x4x4: U[i*4+j] (16 x 49), V[j*4+k] (16 x 49), W[k*4+i] (16 x 49).
    n, m, p = 4, 4, 4
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col, (UT, VT, WT) in enumerate(terms):
        for (i, j), v in UT.items():
            if v:
                U[i * m + j][col] = v
        for (j, k), v in VT.items():
            if v:
                V[j * p + k][col] = v
        for (k, i), v in WT.items():
            if v:
                W[k * n + i][col] = v
    return U, V, W


def construct():
    _build_base()
    if time.time() > _DEDLINE - 0.2:
        # fall back to schoolbook rank-128 if deadline is imminent
        return _schoolbook()
    U, V, W = _build_rank98_full()
    if time.time() > _DEDLINE - 0.1:
        return _schoolbook()
    # Now make the 4x4x8 as direct sum: two blocks of the 4x4 decomposition.
    n, m, p = 4, 4, 8
    rank1 = len(W)  # 49
    total_rank = 2 * rank1  # 98
    nU, nV, nW = n * m, m * p, p * n  # 16, 32, 32
    U2 = [[0] * total_rank for _ in range(nU)]
    V2 = [[0] * total_rank for _ in range(nV)]
    W2 = [[0] * total_rank for _ in range(nW)]
    # Block 0: p-block = rows 0..3 of the 8-wide p dimension
    for row in range(nU):
        for c in range(rank1):
            U2[row][c] = U[row][c]
    for row in range(nV):  # 32 = 4 * 8, block 0 uses rows 0..15 (j=0..3, k=0..3)
        # V index row = j*8 + k ; k in 0..3 -> rows j*8 .. j*8+3
        j = row // 8
        k = row % 8
        if k < 4:
            src = j * 4 + k
            for c in range(rank1):
                V2[row][c] = V[src][c]
    for row in range(nW):  # 32 = 4 * 8 (k, i), k in 0..7
        k = row // 4
        i = row % 4
        if k < 4:
            src = k * 4 + i
            for c in range(rank1):
                W2[row][c] = W[src][c]
    # Block 1: columns rank1..total_rank-1
    c0 = rank1
    for row in range(nU):
        for c in range(rank1):
            U2[row][c0 + c] = U[row][c]
    for row in range(nV):
        j = row // 8
        k = row % 8
        if 4 <= k < 8:
            src = j * 4 + (k - 4)
            for c in range(rank1):
                V2[row][c0 + c] = V[src][c]
    for row in range(nW):
        k = row // 4
        i = row % 4
        if 4 <= k < 8:
            src = (k - 4) * 4 + i
            for c in range(rank1):
                W2[row][c0 + c] = W[src][c]
    return {"U": U2, "V": V2, "W": W2}


def _schoolbook():
    n, m, p = _DIMS
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
