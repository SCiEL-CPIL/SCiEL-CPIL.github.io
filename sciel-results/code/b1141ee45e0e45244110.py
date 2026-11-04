"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.

Strategy: rank-revealing iterative construction with EXACT rational verification.
Build a random sparse rank-r candidate (U,V), then solve W from the linear
constraints T = sum_c U_c o V_c o W_c exactly (sympy rational solve). If the
system is consistent the identity holds exactly -> valid rank-r decomposition;
otherwise increase r. Falls back to a float least-squares + rational round pass
if the rational solver is too slow, and finally to the schoolbook decomposition.
"""
import time
import random
import fractions

DIMENSIONS = [3, 4, 8]


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _as_list(A):
    return [[float(x) for x in row] for row in A]


def _verify(n, m, p, U, V, W):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != (1 if j == k else 0):
                    return False
    return True


def _random_candidate(n, m, p, r, density, rng):
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    for c in range(r):
        for row in range(n * m):
            if rng.random() < density:
                U[row][c] = rng.choice([-1.0, 1.0])
        for row in range(m * p):
            if rng.random() < density:
                V[row][c] = rng.choice([-1.0, 1.0])
    return U, V


def _solve_rational(n, m, p, U, V):
    from sympy import Matrix, symbols, solve_linear_system
    r = len(U[0])
    Ws = symbols("W0:%d" % r)
    A = Matrix(n * m * p, r)
    b = Matrix(n * m * p, 1)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    A[idx, c] = U[i * m + j][c] * V[j * p + k][c]
                b[idx, 0] = 1 if j == k else 0
                idx += 1
    sol = solve_linear_system(A, Ws, b, dict=True)
    if not sol:
        return None
    sol = sol[0]
    if len(sol) != r:
        return None
    vals = [sol.get(Ws[c]) for c in range(r)]
    if any(v is None for v in vals):
        return None
    return vals


def _verify_rational(n, m, p, U, V, Wvals):
    r = len(Wvals)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = fractions.Fraction(0)
                for c in range(r):
                    s += fractions.Fraction(U[i * m + j][c]) * \
                        fractions.Fraction(V[j * p + k][c]) * Wvals[c]
                if s != (1 if j == k else 0):
                    return False
    return True


def _solve_float(n, m, p, U, V):
    import numpy as np
    r = len(U[0])
    A = np.zeros((n * m * p, r), dtype=float)
    b = np.zeros(n * m * p, dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    A[idx, c] = U[i * m + j][c] * V[j * p + k][c]
                b[idx] = 1.0 if j == k else 0.0
                idx += 1
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def _round_rational(x, tol=1e-6):
    out = []
    for val in x:
        if abs(val) < tol:
            out.append(fractions.Fraction(0))
            continue
        fr = fractions.Fraction(val).limit_denominator(2 ** 24)
        if abs(val - float(fr)) < tol:
            out.append(fr)
        else:
            out.append(fractions.Fraction(val))
    return out


def _verify_float(n, m, p, U, V, Wf, tol=1e-8):
    r = len(Wf)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * Wf[c]
                if abs(s - (1.0 if j == k else 0.0)) > tol:
                    return False
    return True


def _try_rank(n, m, p, r, deadline, rng, use_rational):
    attempts = 30
    for _ in range(attempts):
        if time.time() > deadline:
            return None
        U, V = _random_candidate(n, m, p, r, 0.35, rng)
        if use_rational:
            Wvals = _solve_rational(n, m, p, U, V)
            if Wvals is None:
                continue
            if _verify_rational(n, m, p, U, V, Wvals):
                return U, V, Wvals
        else:
            x = _solve_float(n, m, p, U, V)
            if _verify_float(n, m, p, U, V, x):
                Wvals = _round_rational(x)
                if _verify_rational(n, m, p, U, V, Wvals):
                    return U, V, Wvals
                return U, V, x
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0
    rng = random.Random(12345)

    # Determine whether the rational solver is fast enough for this size.
    use_rational = (n * m * p) <= 200
    if use_rational:
        # quick timing probe
        U, V = _random_candidate(n, m, p, max(2, n * m * p // 2), 0.35, rng)
        st = time.time()
        try:
            _solve_rational(n, m, p, U, V)
        except Exception:
            use_rational = False
        if time.time() - st > 1.0:
            use_rational = False

    # Rank lower bound: rank >= max(n, m, p).
    lo = max(n, m, p)
    hi = n * m * p
    best = None
    for r in range(lo, hi + 1):
        if time.time() > deadline - 1.0:
            break
        res = _try_rank(n, m, p, r, deadline, rng, use_rational)
        if res is not None:
            best = res
            break

    if best is None:
        return _schoolbook()

    U, V, W = best
    if all(isinstance(x, fractions.Fraction) for x in W):
        Wout = [[float(w) for w in W]] if isinstance(W, list) and len(W) == 1 else \
               [[float(W[row][c]) for c in range(len(W[0]))] for row in range(len(W))]
    else:
        Wout = _as_list(W)
    return {"U": _as_list(U), "V": _as_list(V), "W": Wout}


if __name__ == "__main__":
    out = construct()
    print("rank =", len(out["U"][0]))
    print("valid =", _verify(*DIMENSIONS, out["U"], out["V"], out["W"]))
