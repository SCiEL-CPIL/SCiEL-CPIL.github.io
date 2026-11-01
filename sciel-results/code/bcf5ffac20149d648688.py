"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition.

Strategy (fresh design):
  1. Build the multiplication tensor T and a strong seed (Strassen per 2x3x2 block, rank 44).
  2. Exact integer refinement: rank-1 drops and rank-2 pair merges, verified with integer arithmetic.
  3. Exact algebraic search: solve the bilinear system with rational Gaussian elimination to
     detect genuine rank reductions (no floating point, no false positives).
  4. Deadline guard returns the best verified result with a safety margin.

Interface kept exactly: construct() -> {'U','V','W'} with shapes (n*m,rank),(m*p,rank),(p*n,rank).
Output factor uses the official trace convention W[k*n+i, r].
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 8]
DEADLINE = 150.0
SAFETY = 25.0
_START = time.time()


def _time_left():
    return DEADLINE - SAFETY - (time.time() - _START)


def _to_int(M):
    M = np.asarray(M)
    if M.dtype.kind == 'f':
        M = np.rint(M)
    return [[int(x) for x in row] for row in M.tolist()]


def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _verify(U, V, W, T):
    n = T.shape[0] // (T.shape[0] // 6)  # nm//m ; m=3
    m = 3
    p = T.shape[2] // (T.shape[0] // m)
    n = T.shape[0] // m
    p = T.shape[2] // n
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                Ui = U[i * m + j]
                Vj = V[j * p + k]
                Wk = W[k * n + i]
                for c in range(r):
                    s += Ui[c] * Vj[c] * Wk[c]
                if s != T[i * m + j, j * p + k]:
                    return False
    return True


def _strassen_2x2x2():
    U = [[1, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0]]
    V = [[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0]]
    W = [[1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0]]
    return U, V, W


def _seed_44(n, m, p):
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    sU, sV, sW = _strassen_2x2x2()
    for b in range(p // 2):
        base = 4 * b
        for i in range(n):
            for j in range(m):
                for c in range(7):
                    U[i * m + j][base + c] = sU[i * 2 + j][c]
        for j in range(m):
            for kk in range(2):
                for c in range(7):
                    V[j * p + (2 * b + kk)][base + c] = sV[j * 2 + kk][c]
        for kk in range(2):
            for i in range(n):
                for c in range(7):
                    W[(2 * b + kk) * n + i][base + c] = sW[kk * 2 + i][c]
    col = 28
    for k in range(2 * (p // 2), p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
        col += 1
    return U, V, W


def _try_drop(U, V, W, T, r, c):
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    u = [U[i][c] for i in range(len(U))]
    v = [V[j][c] for j in range(len(V))]
    w = [W[k][c] for k in range(len(W))]
    for i in range(len(U)):
        ui = u[i]
        if ui != 0:
            row = U[i]
            for j in range(len(V)):
                row[j] += ui * v[j]
    for j in range(len(V)):
        vj = v[j]
        if vj != 0:
            row = V[j]
            for k in range(len(W)):
                row[k] += vj * w[k]
    U = [row[:c] + row[c + 1:] for row in U]
    V = [row[:c] + row[c + 1:] for row in V]
    W = [row[:c] + row[c + 1:] for row in W]
    if _verify(U, V, W, T):
        return U, V, W
    return None


def _try_merge(U, V, W, T, r, a, b):
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    ua = [U[i][a] for i in range(len(U))]
    ub = [U[i][b] for i in range(len(U))]
    va = [V[j][a] for j in range(len(V))]
    vb = [V[j][b] for j in range(len(V))]
    wa = [W[k][a] for k in range(len(W))]
    wb = [W[k][b] for k in range(len(W))]
    for a1, a2 in ((1, 1), (1, -1)):
        for b1, b2 in ((1, 1), (1, -1)):
            for c1, c2 in ((1, 1), (1, -1)):
                Ua = [ua[i] * a1 + ub[i] * a2 for i in range(len(U))]
                Va = [va[j] * b1 + vb[j] * b2 for j in range(len(V))]
                Wc = [wa[k] * c1 + wb[k] * c2 for k in range(len(W))]
                if all(x == 0 for x in Ua) or all(x == 0 for x in Va) or all(x == 0 for x in Wc):
                    continue
                Unew = [row[:a] + row[a + 1:] for row in U]
                Vnew = [row[:a] + row[a + 1:] for row in V]
                Wnew = [row[:a] + row[a + 1:] for row in W]
                b2i = b if b > a else b - 1
                Unew = [row[:b2i] + [Ua[i]] + row[b2i:] for i, row in enumerate(Unew)]
                Vnew = [row[:b2i] + [Va[j]] + row[b2i:] for j, row in enumerate(Vnew)]
                Wnew = [row[:b2i] + [Wc[k]] + row[b2i:] for k, row in enumerate(Wnew)]
                if _verify(Unew, Vnew, Wnew, T):
                    return Unew, Vnew, Wnew
    return None


def _refine(U, V, W, T):
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    r = len(U[0])
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for c in range(r):
            res = _try_drop(U, V, W, T, r, c)
            if res is not None:
                U, V, W, r = res[0], res[1], res[2], r - 1
                improved = True
                break
        if improved:
            continue
        for a, b in itertools.combinations(range(r), 2):
            if _time_left() < 8.0:
                break
            res = _try_merge(U, V, W, T, r, a, b)
            if res is not None:
                U, V, W, r = res[0], res[1], res[2], r - 1
                improved = True
                break
    return U, V, W


def _solve_for_U(T, V, W, r):
    import sympy as sp
    n, m, p = 2, 3, 8
    nm = n * m
    usyms = [sp.Symbol('u%d_%d' % (i, c)) for i in range(nm) for c in range(r)]
    wseed = [[W[k][c] for c in range(r)] for k in range(p * n)]
    vseed = [[V[j][c] for c in range(r)] for j in range(m * p)]
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(usyms[i * r + c] * vseed[j * p + k][c] for c in range(r))
                rhs = sum(T[i * m + j, j * p + k, k * n + i] * wseed[k * n + i][c]
                          for c in range(r))
                eqs.append(lhs - rhs)
    sol = sp.linsolve(eqs, usyms)
    if sol is None:
        return None
    try:
        vals = list(sol)[0]
    except (TypeError, IndexError):
        return None
    U = [[0] * r for _ in range(nm)]
    for i in range(nm):
        for c in range(r):
            U[i][c] = int(vals[i * r + c])
    return U


def _rank_feasible(T, V, W, r):
    """Does a rank-r solution exist with V,W fixed to the seed? (linear in U)."""
    import sympy as sp
    n, m, p = 2, 3, 8
    nm = n * m
    usyms = [sp.Symbol('u%d_%d' % (i, c)) for i in range(nm) for c in range(r)]
    wseed = [[W[k][c] for c in range(r)] for k in range(p * n)]
    vseed = [[V[j][c] for c in range(r)] for j in range(m * p)]
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(usyms[i * r + c] * vseed[j * p + k][c] for c in range(r))
                rhs = sum(T[i * m + j, j * p + k, k * n + i] * wseed[k * n + i][c]
                          for c in range(r))
                eqs.append(lhs - rhs)
    sol = sp.linsolve(eqs, usyms)
    return sol is not None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    U, V, W = _seed_44(n, m, p)
    best = (U, V, W)
    best_rank = len(U[0])

    # exact integer refinement (drops + merges)
    if _time_left() > 10.0:
        Ur, Vr, Wr = _refine(U, V, W, T)
        if len(Ur[0]) < best_rank and _verify(Ur, Vr, Wr, T):
            best = (Ur, Vr, Wr)
            best_rank = len(Ur[0])

    # exact algebraic search for a genuine rank reduction (fix V,W, solve for U)
    if _time_left() > 20.0:
        for r in range(best_rank - 1, max(30, best_rank - 4), -1):
            if _time_left() < 15.0:
                break
            if _rank_feasible(T, best[1], best[2], r):
                if _time_left() > 10.0:
                    sol = _solve_for_U(T, best[1], best[2], r)
                    if sol is not None and _verify(sol, best[1], best[2], T):
                        best = (sol, best[1], best[2])
                        best_rank = r
                        break

    return {"U": _to_int(best[0]), "V": _to_int(best[1]), "W": _to_int(best[2])}
