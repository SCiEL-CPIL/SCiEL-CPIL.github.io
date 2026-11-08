"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
START_TIME = time.time()
DEADLINE = 150.0
SAFETY = 8.0


def _left():
    return DEADLINE - SAFETY - (time.time() - START_TIME)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_X(n, m, p):
    X = np.zeros((n * p, n * m, m * p), dtype=float)
    for i in range(n):
        for k in range(p):
            out = k * n + i
            for j in range(m):
                A = i * m + j
                B = j * p + k
                X[out, A, B] = 1.0
    return X


def _round_half(M):
    return np.round(2.0 * M) / 2.0


def _float_ok(X, U, V, W, tol=1e-4):
    recon = np.einsum('or,ar,br->oab', W, U, V, optimize=True)
    return np.max(np.abs(recon - X)) < tol


def _exact_ok(U, V, W, n, m, p):
    r = U.shape[1]
    Uf = [[Fraction(int(round(2 * x))) / 2 for x in row] for row in U]
    Vf = [[Fraction(int(round(2 * x))) / 2 for x in row] for row in V]
    Wf = [[Fraction(int(round(2 * x))) / 2 for x in row] for row in W]
    count = 0
    for out in range(n * p):
        i = out % n
        k = out // n
        for A in range(n * m):
            ip = A // m
            j = A % m
            for B in range(m * p):
                jp = B // p
                kp = B % p
                expected = 1 if (i == ip and k == kp and j == jp) else 0
                s = Fraction(0)
                for rr in range(r):
                    s += Wf[out][rr] * Uf[A][rr] * Vf[B][rr]
                if s != expected:
                    return False
                count += 1
                if count % 5000 == 0 and _left() < 3:
                    return False  # bail out, treat as not-verified
    return True


def _als(X, r, iters, rng):
    I, J, K = X.shape
    U = rng.standard_normal((I, r))
    V = rng.standard_normal((J, r))
    W = rng.standard_normal((K, r))
    X0 = X.reshape(I, J * K)
    X1 = np.transpose(X, (1, 0, 2)).reshape(J, I * K)
    X2 = np.transpose(X, (2, 0, 1)).reshape(K, I * J)
    for _ in range(iters):
        VW = np.kron(V, W)
        U = np.linalg.lstsq(VW, X0.T, rcond=None)[0].T
        UW = np.kron(U, W)
        V = np.linalg.lstsq(UW, X1.T, rcond=None)[0].T
        UV = np.kron(U, V)
        W = np.linalg.lstsq(UV, X2.T, rcond=None)[0].T
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    X = _build_X(n, m, p)
    best = _schoolbook(n, m, p)

    # SVD matricization lower bound on CP rank
    lb = 1
    for mat in (X.reshape(n * p, -1),
                np.transpose(X, (1, 0, 2)).reshape(n * m, -1),
                np.transpose(X, (2, 0, 1)).reshape(m * p, -1)):
        s = np.linalg.svd(mat, compute_uv=False)
        lb = max(lb, int(np.sum(s > 1e-8)))

    rng = np.random.default_rng(0)
    r = max(lb, 1)
    max_r = 60
    while r <= max_r and _left() > 6:
        n_inits = 5 if r <= lb + 5 else 2
        if _left() < 20:
            n_inits = 1
        for _ in range(n_inits):
            if _left() < 6:
                break
            iters = 25 if _left() > 25 else 10
            U, V, W = _als(X, r, iters, rng)
            Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
            if _float_ok(X, Ur, Vr, Wr, tol=1e-4):
                if _exact_ok(Ur, Vr, Wr, n, m, p):
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
        r += 1
    return best
