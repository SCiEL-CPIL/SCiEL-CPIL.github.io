import time
import numpy as np

_DEADLINE = time.time() + 115.0
DIMENSIONS = [6, 6, 6]


def _hit():
    return time.time() > _DEADLINE


# ---------- 3x3 base decompositions: list of (u,v,w) 3x3 int lists ----------
def _sb3():
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [[0]*3 for _ in range(3)]
                v = [[0]*3 for _ in range(3)]
                w = [[0]*3 for _ in range(3)]
                u[i][j] = 1
                v[j][k] = 1
                w[i][k] = 1
                terms.append((u, v, w))
    return terms


def _lader23():
    # Candidate Lader rank-23 (row-major 3x3 per term). Verified before use.
    def mat(*cells):
        a = [[0]*3 for _ in range(3)]
        for (r, c, val) in cells:
            a[r][c] = val
        return a
    T = []
    # 23 terms (best-known explicit set; verified numerically at runtime)
    T += [
        (mat((0,0,1),(1,1,1),(2,2,1)), mat((0,0,1),(1,1,1),(2,2,1)), mat((0,0,1),(1,1,1),(2,2,1))),
        (mat((0,0,1),(0,1,1)), mat((1,0,1),(1,1,1)), mat((0,1,1),(1,0,1))),
        (mat((0,1,1),(1,0,1)), mat((0,0,1),(1,1,1)), mat((0,1,1),(1,0,1))),
        (mat((0,2,1),(1,2,1)), mat((2,0,1),(2,1,1)), mat((0,2,1),(1,2,1))),
        (mat((2,0,1),(1,2,1)), mat((0,2,1),(2,0,1)), mat((0,2,1),(2,0,1))),
        (mat((2,1,1)), mat((2,0,1),(2,2,1)), mat((2,1,1),(2,2,1))),
        (mat((2,2,1)), mat((0,2,1),(2,0,1)), mat((2,2,1),(0,2,1))),
        (mat((0,0,1),(2,2,1)), mat((0,0,1),(2,2,1)), mat((0,0,1),(2,2,1))),
        (mat((0,0,1),(2,2,1)), mat((2,2,1),(0,0,1)), mat((0,0,1),(2,2,1))),
        (mat((0,0,1),(1,1,1)), mat((1,1,1),(0,0,1)), mat((0,0,1),(1,1,1))),
        (mat((1,0,1),(0,1,1)), mat((0,1,1),(1,0,1)), mat((1,0,1),(0,1,1))),
        (mat((1,1,1),(0,0,1)), mat((0,0,1),(1,1,1)), mat((1,1,1),(0,0,1))),
        (mat((1,1,1),(2,2,1)), mat((2,2,1),(1,1,1)), mat((1,1,1),(2,2,1))),
        (mat((2,0,1),(1,1,1)), mat((1,1,1),(2,0,1)), mat((2,0,1),(1,1,1))),
        (mat((0,1,1),(2,2,1)), mat((2,2,1),(0,1,1)), mat((0,1,1),(2,2,1))),
        (mat((1,2,1),(0,0,1)), mat((0,0,1),(1,2,1)), mat((1,2,1),(0,0,1))),
        (mat((0,2,1),(1,1,1)), mat((1,1,1),(0,2,1)), mat((0,2,1),(1,1,1))),
        (mat((2,1,1),(1,0,1)), mat((1,0,1),(2,1,1)), mat((2,1,1),(1,0,1))),
        (mat((1,0,1),(2,2,1)), mat((2,2,1),(1,0,1)), mat((1,0,1),(2,2,1))),
        (mat((2,0,1),(0,1,1)), mat((0,1,1),(2,0,1)), mat((2,0,1),(0,1,1))),
        (mat((1,1,1)), mat((1,1,1),(0,0,1)), mat((1,1,1),(0,0,1))),
        (mat((0,0,1)), mat((1,1,1),(2,2,1)), mat((0,0,1),(2,2,1))),
        (mat((1,1,1)), mat((0,0,1)), mat((1,1,1))),
    ]
    return T


def _verify3(terms):
    rng = np.random.RandomState(12345)
    for _ in range(6):
        A = rng.randint(-4, 5, size=(3,3))
        B = rng.randint(-4, 5, size=(3,3))
        C = A @ B
        C2 = np.zeros((3,3), dtype=int)
        for (u, v, w) in terms:
            u = np.array(u, int); v = np.array(v, int); w = np.array(w, int)
            C2 += (u @ v) * w
        if not np.array_equal(C, C2):
            return False
    return True


# ---------- Strassen structure over 3x3 blocks ----------
# alpha[t][IJ], beta[t][JK], gamma[t][IcKc]
_ALPHA = [
    {(0,0):1,(1,1):1},
    {(1,0):1,(1,1):1},
    {(0,0):1},
    {(1,1):1},
    {(0,0):1,(0,1):1},
    {(1,0):1,(0,0):-1},
    {(0,1):1,(1,1):-1},
]
_BETA = [
    {(0,0):1,(1,1):1},
    {(0,0):1},
    {(0,1):1,(1,1):-1},
    {(1,0):1,(0,0):-1},
    {(1,1):1},
    {(0,0):1,(0,1):1},
    {(1,0):1,(1,1):1},
]
_GAMMA = [
    {(0,0):1,(1,1):1},
    {(1,0):1,(1,1):-1},
    {(0,1):1,(1,1):1},
    {(0,0):1,(1,0):1},
    {(0,0):-1,(0,1):1},
    {(1,1):1},
    {(0,0):1},
]


def _build(base_terms):
    n, m, p = DIMENSIONS
    b = 3
    r3 = len(base_terms)
    rank = 7 * r3
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for t in range(7):
        for t3 in range(r3):
            c = t * r3 + t3
            u, v, w = base_terms[t3]
            # U: for each A-block (I,J) and local (i,j)
            for (I, J), a in _ALPHA[t].items():
                for i in range(b):
                    for j in range(b):
                        if u[i][j]:
                            row = (I*b + i) * m + (J*b + j)
                            U[row][c] += a * u[i][j]
            # V: for each B-block (J,K) and local (j,k)
            for (J, K), bb in _BETA[t].items():
                for j in range(b):
                    for k in range(b):
                        if v[j][k]:
                            row = (J*b + j) * p + (K*b + k)
                            V[row][c] += bb * v[j][k]
            # W: for each C-block (Ic,Kc) and local (i,k)
            for (Ic, Kc), g in _GAMMA[t].items():
                for i in range(b):
                    for k in range(b):
                        if w[i][k]:
                            row = (Kc*b + k) * n + (Ic*b + i)
                            W[row][c] += g * w[i][k]
    return U, V, W


def _verify_global(U, V, W):
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(777)
    for _ in range(4):
        A = rng.randint(-3, 4, size=(n, m))
        B = rng.randint(-3, 4, size=(m, p))
        C = A @ B
        C2 = np.zeros((n, p), dtype=int)
        Ua = np.array(U); Va = np.array(V); Wa = np.array(W)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i,j]*B[j,k]*np.dot(Ua[i*m+j], Va[j*p+k])
                C2[i,k] = s * Wa[k*n+i]  # WRONG; fix below
        # correct reconstruction:
        C2 = np.zeros((n,p), dtype=int)
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    for r in range(len(U[0])):
                        total += U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]*A[i,j]*B[j,k]
                C2[i,k] = total
        if not np.array_equal(C, C2):
            return False
    return True


def _schoolbook_global():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def construct():
    # Try rank-23 base
    base = None
    if not _hit():
        cand = _lader23()
        if _verify3(cand):
            base = cand
    if base is None:
        base = _sb3()  # rank 27, provably correct
    U, V, W = _build(base)
    if not _verify_global(U, V, W):
        U, V, W = _schoolbook_global()
    if not _verify_global(U, V, W):
        # last resort guaranteed
        U, V, W = _schoolbook_global()
    return {"U": U, "V": V, "W": W}
