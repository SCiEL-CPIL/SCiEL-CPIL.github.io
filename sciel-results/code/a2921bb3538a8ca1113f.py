import numpy as np
import time
import sys
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

# Precompute index triples once (avoid per-call list builds).
_N, _M, _P = DIMENSIONS
_TRIPLES = np.array([(i, j, k) for i in range(_N) for j in range(_M) for k in range(_P)], dtype=int)
_I_IDX = _TRIPLES[:, 0]
_J_IDX = _TRIPLES[:, 1]
_K_IDX = _TRIPLES[:, 2]
_U_IDX = _I_IDX * _M + _J_IDX          # row in U
_V_IDX = _J_IDX * _P + _K_IDX          # row in V
_W_IDX = _K_IDX * _N + _I_IDX          # row in W
_NTRIP = len(_TRIPLES)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _residual_fast(x, n, m, p, rank):
    """Vectorized residual: for each (i,j,k), sum_r U*V*W - 1.  O(n*m*p*rank)."""
    size_U = n * m * rank
    size_V = m * p * rank
    U = x[:size_U].reshape(n * m, rank)
    V = x[size_U:size_U + size_V].reshape(m * p, rank)
    W = x[size_U + size_V:].reshape(p * n, rank)
    vals = np.einsum('ir,sr,tr->t', U[_U_IDX], V[_V_IDX], W[_W_IDX])
    return vals - 1.0


def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic (handles int / half-int / float)."""
    Uf = np.asarray(U)
    Vf = np.asarray(V)
    Wf = np.asarray(W)
    for t in range(_NTRIP):
        u = Uf[_U_IDX[t]]
        v = Vf[_V_IDX[t]]
        w = Wf[_W_IDX[t]]
        acc = Fraction(0)
        for a, b, c in zip(u, v, w):
            acc += Fraction(a) * Fraction(b) * Fraction(c)
        if acc != 1:
            return False
    return True


def _verify_float(U, V, W, n, m, p, tol=1e-8):
    Uf = np.asarray(U, dtype=float)
    Vf = np.asarray(V, dtype=float)
    Wf = np.asarray(W, dtype=float)
    vals = np.einsum('ir,sr,tr->t', Uf[_U_IDX], Vf[_V_IDX], Wf[_W_IDX])
    return bool(np.all(np.abs(vals - 1.0) <= tol))


def _try_round(U, V, W, n, m, p, tol):
    """Round to half-integers; return (ok, U, V, W) with exact verification."""
    Uf = U.flatten(); Vf = V.flatten(); Wf = W.flatten()
    dU = np.max(np.abs(Uf - np.round(2 * Uf) / 2.0))
    dV = np.max(np.abs(Vf - np.round(2 * Vf) / 2.0))
    dW = np.max(np.abs(Wf - np.round(2 * Wf) / 2.0))
    if max(dU, dV, dW) >= tol:
        return False, None, None, None
    Ur = np.round(2 * Uf) / 2.0
    Vr = np.round(2 * Vf) / 2.0
    Wr = np.round(2 * Wf) / 2.0
    U2 = Ur.reshape(U.shape); V2 = Vr.reshape(V.shape); W2 = Wr.reshape(W.shape)
    if _verify_exact(U2, V2, W2, n, m, p):
        return True, U2, V2, W2
    return False, None, None, None


def _check_and_round(U, V, W, n, m, p):
    """Try progressively wider half-integer rounding tolerances."""
    for tol in (1e-4, 1e-3, 1e-2, 5e-2, 0.2):
        ok, Ur, Vr, Wr = _try_round(U, V, W, n, m, p, tol)
        if ok:
            return True, Ur, Vr, Wr
    return False, None, None, None


def _structured_init(n, m, p, rank, seed=0):
    rng = np.random.RandomState(seed)
    size_U = n * m * rank
    size_V = m * p * rank
    total_size = size_U + size_V + p * n * rank
    x0 = np.zeros(total_size)
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    rng.shuffle(all_triples)
    for r in range(rank):
        start = r * (len(all_triples) // rank)
        end = (r + 1) * (len(all_triples) // rank) if r < rank - 1 else len(all_triples)
        for (i, j, k) in all_triples[start:end]:
            x0[i * m + j + r * n * m] = 1.0
            x0[(n * m) + (j * p + k) + r * m * p] = 1.0
            x0[(n * m + m * p) + (k * n + i) + r * p * n] = 1.0
    x0 += rng.normal(0, 0.05, total_size)
    return x0


def _random_init(n, m, p, rank, seed=0, scale=0.5):
    rng = np.random.RandomState(seed)
    total_size = (n * m + m * p + p * n) * rank
    return rng.normal(0, scale, total_size)


def _ladder_init(n, m, p, rank, seed=0):
    rng = np.random.RandomState(seed)
    size_U = n * m * rank
    size_V = m * p * rank
    total_size = size_U + size_V + p * n * rank
    x0 = np.zeros(total_size)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(min(3, p)):
                if col < rank:
                    x0[i * m + j + col * n * m] = 1.0
                    x0[(n * m) + (j * p + k) + col * m * p] = 1.0
                    x0[(n * m + m * p) + (k * n + i) + col * p * n] = 1.0
                    col += 1
    for r in range(col, rank):
        for i in range(n):
            for j in range(m):
                x0[i * m + j + r * n * m] = rng.normal(0, 0.3)
        for k in range(p):
            for j in range(m):
                x0[(n * m) + (j * p + k) + r * m * p] = rng.normal(0, 0.3)
        for k in range(p):
            for i in range(n):
                x0[(n * m + m * p) + (k * n + i) + r * p * n] = rng.normal(0, 0.3)
    x0 += rng.normal(0, 0.02, total_size)
    return x0


def _solve_rank(n, m, p, rank, max_restarts, deadline):
    size_U = n * m * rank
    size_V = m * p * rank
    total_size = size_U + size_V + p * n * rank
    func = lambda x: _residual_fast(x, n, m, p, rank)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        if restart == 0:
            x0 = _ladder_init(n, m, p, rank, seed=42 + restart)
        elif restart % 3 == 1:
            x0 = _structured_init(n, m, p, rank, seed=restart * 1000 + 42)
        else:
            x0 = _random_init(n, m, p, rank, seed=restart * 1000 + 137,
                              scale=0.3 + 0.2 * ((restart % 4) / 3.0))
        try:
            res = least_squares(func, x0, method='lm', xtol=1e-13, ftol=1e-13,
                                gtol=1e-13, max_nfev=4000)
        except Exception:
            continue
        if res.cost < 1e-9:
            x_sol = res.x
            U = x_sol[:size_U].reshape(n * m, rank)
            V = x_sol[size_U:size_U + size_V].reshape(m * p, rank)
            W = x_sol[size_U + size_V:].reshape(p * n, rank)
            ok, Ur, Vr, Wr = _check_and_round(U, V, W, n, m, p)
            if ok:
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
            # High-precision polish pass (tighter tol, more iters, exact local solve).
            try:
                res2 = least_squares(func, x_sol, method='lm', xtol=1e-15, ftol=1e-15,
                                     gtol=1e-15, max_nfev=3000)
                if res2.cost < 1e-11:
                    U2 = res2.x[:size_U].reshape(n * m, rank)
                    V2 = res2.x[size_U:size_U + size_V].reshape(m * p, rank)
                    W2 = res2.x[size_U + size_V:].reshape(p * n, rank)
                    ok2, Ur2, Vr2, Wr2 = _check_and_round(U2, V2, W2, n, m, p)
                    if ok2:
                        return {"U": Ur2.tolist(), "V": Vr2.tolist(), "W": Wr2.tolist()}
            except Exception:
                pass
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 112  # safety margin before the 150 s kill

    # Rank 3x3 x 3x4 is at most 36 (schoolbook); known 3x3x3 rank is 23,
    # so a split construction gives <= 32, but shared structure may do better.
    # We prioritize ranks 23-27 as they are the most likely minimal or near-minimal.
    ranks_to_try = list(range(23, 28)) + list(range(16, 23)) + list(range(28, 37))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        if 23 <= rank <= 27:
            max_restarts = 40
        elif rank <= 22:
            max_restarts = 5
        else:
            max_restarts = 10
        result = _solve_rank(n, m, p, rank, max_restarts, deadline)
        if result is not None:
            return result

    return _schoolbook(n, m, p)
