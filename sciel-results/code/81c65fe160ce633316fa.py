import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
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

def _verify(U, V, W, tol=1e-5):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = float(np.dot(Ua[a, :], Va[b, :] * Wa[c, :]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_int(M, tol=1e-4):
    for row in M:
        for x in row:
            if abs(x * 2.0 - round(x * 2.0)) > tol:
                return False
    return True

def _round_half(M):
    return [[int(round(x * 2.0)) / 2.0 for x in row] for row in M]

def _solve_W(U, V, n, m, p, r):
    pn = p * n
    A = np.zeros((n * m * p, pn * r))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = k * n + i
                for col in range(r):
                    A[eq, row * r + col] = U[i * m + j][col] * V[j * p + k][col]
                eq += 1
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    W = sol.reshape(pn, r)
    return W

def _solve_V(U, W, n, m, p, r):
    mp = m * p
    A = np.zeros((n * m * p, mp * r))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = j * p + k
                for col in range(r):
                    A[eq, row * r + col] = U[i * m + j][col] * W[k * n + i][col]
                eq += 1
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    V = sol.reshape(mp, r)
    return V

def _solve_U(V, W, n, m, p, r):
    nm = n * m
    A = np.zeros((n * m * p, nm * r))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m + j
                for col in range(r):
                    A[eq, row * r + col] = V[j * p + k][col] * W[k * n + i][col]
                eq += 1
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    U = sol.reshape(nm, r)
    return U

def _rand_half(rows, r):
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [[random.choice(vals) for _ in range(r)] for _ in range(rows)]

def construct():
    deadline = time.time() + 140.0  # explicit guard: 10 s safety margin before 150 s kill
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    best = _schoolbook()
    best_rank = base_rank

    random.seed(20240517)

    lo = max(1, base_rank - 1)
    candidates = list(range(lo, base_rank))
    max_ranks = 40
    if len(candidates) > max_ranks:
        idx = sorted(random.sample(range(len(candidates)), max_ranks))
        candidates = [candidates[i] for i in idx]
    candidates = sorted(set(candidates))

    for r in candidates:
        if time.time() > deadline - 8.0:
            break
        for attempt in range(2):
            if time.time() > deadline - 8.0:
                break
            U = _rand_half(n * m, r)
            V = _rand_half(m * p, r)
            W = _rand_half(p * n, r)
            found = False
            for _it in range(30):
                if time.time() > deadline - 8.0:
                    break
                W = _solve_W(U, V, n, m, p, r)
                V = _solve_V(U, W, n, m, p, r)
                U = _solve_U(V, W, n, m, p, r)
                Ur = _round_half(U); Vr = _round_half(V); Wr = _round_half(W)
                if _verify(Ur, Vr, Wr) and _is_half_int(Ur) and _is_half_int(Vr) and _is_half_int(Wr):
                    if r < best_rank:
                        best = {"U": Ur, "V": Vr, "W": Wr}
                        best_rank = r
                    found = True
                    break
            if found:
                break
    return best
# EVOLVE-BLOCK-END
