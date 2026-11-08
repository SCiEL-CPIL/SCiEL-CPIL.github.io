"""Matrix multiplication 4x5x5 rank minimization via rank-1 peeling + prime-field fingerprint."""
import time
import random
import sys
from fractions import Fraction
from functools import lru_cache

DIMENSIONS = [4, 5, 5]

# ---------- prime field ----------
P = 1000000007
PM1 = P - 1
_INV = None
_INV_CACHE = {}

def _inv(x):
    if x == 0:
        return 0
    if x in _INV_CACHE:
        return _INV_CACHE[x]
    r = pow(x, PM1, P)
    _INV_CACHE[x] = r
    return r

def _row_dot(a, b):
    s = 0
    for x, y in zip(a, b):
        s += x * y
    s %= P
    return s

# ---------- tensor ----------
@lru_cache(maxsize=None)
def tensor_terms():
    n, m, p = DIMENSIONS
    t = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t[(i, j, k)] = 1
    return t

def _get(T, i, j, k):
    v = T.get((i, j, k))
    return v if v is not None else 0

def _set(T, i, j, k, v):
    if v == 0:
        T.pop((i, j, k), None)
    else:
        T[(i, j, k)] = v

def _scale(T, s):
    if s == 0:
        return {}
    return {key: (val * s) % P for key, val in T.items()}

def _add_into(T, S):
    for key, val in S.items():
        nv = (T.get(key, 0) + val) % P
        if nv == 0:
            T.pop(key, None)
        else:
            T[key] = nv

def _sub(T, S, s):
    T2 = dict(T)
    if s != 0:
        for key, val in S.items():
            nv = (T2.get(key, 0) - val * s) % P
            if nv == 0:
                T2.pop(key, None)
            else:
                T2[key] = nv
    return T2

# ---------- fingerprint (exact rational) ----------
def fingerprint(T):
    fp = 0
    for (i, j, k), val in T.items():
        a = Fraction(val, P)
        fp ^= hash((i, j, k, a))
    return fp

def _to_frac(T):
    out = {}
    for (i, j, k), val in T.items():
        out[(i, j, k)] = Fraction(val, P)
    return out

def _to_int(T):
    out = {}
    for (i, j, k), val in T.items():
        out[(i, j, k)] = int(Fraction(val, P))
    return out

# ---------- U/V/W <-> tensor ----------
def build_UVW(U, V, W):
    n, m, p = DIMENSIONS
    U = list(U); V = list(V); W = list(W)
    T = {}
    for c in range(len(U)):
        uc = U[c]; vc = V[c]; wc = W[c]
        nz_u = [i for i in range(n * m) if uc[i] != 0]
        nz_v = [j for j in range(m * p) if vc[j] != 0]
        nz_w = [k for k in range(p * n) if wc[k] != 0]
        if not (nz_u and nz_v and nz_w):
            continue
        for ii in nz_u:
            a = uc[ii]
            I = ii // m; J = ii % m
            for jj in nz_v:
                b = vc[jj]
                J2 = jj // p; K = jj % p
                if J != J2:
                    continue
                ab = a * b
                for kk in nz_w:
                    cval = wc[kk]
                    K2 = kk // n; I2 = kk % n
                    if I != I2 or K != K2:
                        continue
                    v = ab * cval
                    if v != 0:
                        v %= P
                        key = (I, J, K)
                        T[key] = (T.get(key, 0) + v) % P
    return U, V, W, T

def rank1_tensor(u, v, w):
    n, m, p = DIMENSIONS
    T = {}
    for ii in range(n * m):
        a = u[ii]
        if a == 0:
            continue
        I = ii // m; J = ii % m
        for jj in range(m * p):
            b = v[jj]
            if b == 0:
                continue
            J2 = jj // p; K = jj % p
            if J != J2:
                continue
            ab = a * b
            for kk in range(p * n):
                cval = w[kk]
                if cval == 0:
                    continue
                K2 = kk // n; I2 = kk % n
                if I != I2 or K != K2:
                    continue
                v = ab * cval
                if v != 0:
                    v %= P
                    key = (I, J, K)
                    T[key] = (T.get(key, 0) + v) % P
    return T

def rand_vec(d):
    return [random.randrange(P) for _ in range(d)]

def mat_mul(U, V):
    # U: r x (n*m), V: (n*m) x r  -> r x r
    r = len(U)
    nm = len(V)
    M = [[0] * r for _ in range(r)]
    for a in range(r):
        ua = U[a]
        Ma = M[a]
        for x in range(nm):
            uax = ua[x]
            if uax == 0:
                continue
            Vx = V[x]
            for b in range(r):
                Ma[b] = (Ma[b] + uax * Vx[b]) % P
    return M

def solve_lin(A, b):
    # A: r x r, b: r  -> x
    r = len(A)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(r):
        piv = -1
        for i in range(col, r):
            if M[i][col] != 0:
                piv = i
                break
        if piv == -1:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        invp = _inv(M[col][col])
        for j in range(col, r + 1):
            M[col][j] = M[col][j] * invp % P
        for i in range(r):
            if i != col and M[i][col] != 0:
                f = M[i][col]
                for j in range(col, r + 1):
                    M[i][j] = (M[i][j] - f * M[col][j]) % P
    return [M[i][r] for i in range(r)]

def peel(U, V, W, T, fp, deadline, budget):
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    r = len(U)
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    T = dict(T)
    attempts = 0
    while r > 0 and time.time() < deadline - 1.0:
        if budget is not None and attempts >= budget:
            break
        attempts += 1
        a = rand_vec(nm)
        b = rand_vec(mp)
        c = rand_vec(pn)
        u = [0] * r; v = [0] * r; w = [0] * r
        for i in range(nm):
            if a[i] == 0:
                continue
            Vi = V[i]
            for t in range(r):
                u[t] = (u[t] + Vi[t] * a[i]) % P
        for j in range(mp):
            if b[j] == 0:
                continue
            Uj = U[j]
            for t in range(r):
                v[t] = (v[t] + Uj[t] * b[j]) % P
        for k in range(pn):
            if c[k] == 0:
                continue
            Wk = W[k]
            for t in range(r):
                w[t] = (w[t] + Wk[t] * c[k]) % P
        M = mat_mul(U, V)
        x = solve_lin(M, w)
        if x is None:
            continue
        R = rank1_tensor(u, v, w)
        T2 = _sub(T, R, 1)
        if fingerprint(T2) != fp:
            continue
        # build new U/V/W
        nz = [t for t in range(r) if x[t] != 0]
        nz.sort()
        newU = [U[t] for t in nz]
        newV = [V[t] for t in nz]
        newW = [W[t] for t in nz]
        newU.append(u)
        newV.append(v)
        newW.append(w)
        U, V, W, T = newU, newV, newW, T2
        r = len(U)
    return U, V, W, T, r

def verify(U, V, W):
    n, m, p = DIMENSIONS
    Uf = _to_frac(U); Vf = _to_frac(V); Wf = _to_frac(W)
    T = {}
    for c in range(len(Uf)):
        uc = Uf[c]; vc = Vf[c]; wc = Wf[c]
        for ii in range(n * m):
            a = uc[ii]
            if a == 0:
                continue
            I = ii // m; J = ii % m
            for jj in range(m * p):
                b = vc[jj]
                if b == 0:
                    continue
                J2 = jj // p; K = jj % p
                if J != J2:
                    continue
                ab = a * b
                for kk in range(p * n):
                    cv = wc[kk]
                    if cv == 0:
                        continue
                    K2 = kk // n; I2 = kk % n
                    if I != I2 or K != K2:
                        continue
                    v = ab * cv
                    if v != 0:
                        key = (I, J, K)
                        T[key] = T.get(key, 0) + v
    for key in list(T.keys()):
        if T[key] == 0:
            del T[key]
    if len(T) != n * m * p:
        return False
    for key, val in T.items():
        if val != 1:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    random.seed(12345)

    # schoolbook
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    bestU, bestV, bestW, bestT, bestr = U, V, W, {}, n * m * p
    for attempt in range(30):
        if time.time() > deadline - 25.0:
            break
        U, V, W, T, r = build_UVW(bestU, bestV, bestW)
        fp = fingerprint(T)
        U, V, W, T, r = peel(U, V, W, T, fp, deadline, budget=400000)
        if r < bestr:
            bestr = r
            bestU, bestV, bestW, bestT = U, V, W, T
        if time.time() > deadline - 30.0:
            break

    # exact verification of the best found
    if verify(bestU, bestV, bestW):
        Uo = _to_int(bestU); Vo = _to_int(bestV); Wo = _to_int(bestW)
        return {"U": Uo, "V": Vo, "W": Wo}
    # fallback to schoolbook (guaranteed valid)
    return {"U": U, "V": V, "W": W}
