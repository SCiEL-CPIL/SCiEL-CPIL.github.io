"""Rewritten program: smooth constrained optimization to minimize max/min distance ratio."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
_IU = np.triu_indices(N, k=1)

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START) - 2.0


def _pairwise_sq(P):
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _soft_min(x, beta=20.0):
    return -logsumexp(-x / beta) * beta


def _score(P):
    d2 = _pairwise_sq(P)
    vals = d2[_IU]
    if vals.min() <= 1e-12:
        return np.inf
    return (vals.max() / vals.min()) ** 2


def _objective_penalty(P, beta=20.0, mu=1e4):
    d2 = _pairwise_sq(P)
    vals = d2[_IU]
    obj = -_soft_min(vals, beta)
    over = np.maximum(vals - 1.0, 0.0)
    obj += mu * np.sum(over * over)
    obj += 1e-3 * np.sum(P * P)
    return obj


def _grad_penalty(P, beta=20.0, mu=1e4):
    d2 = _pairwise_sq(P)
    vals = d2[_IU]
    ii, jj = _IU
    z = -vals / beta
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    W = np.zeros((N, N))
    W[ii, jj] = w
    W[jj, ii] = w
    rowsum = W.sum(axis=1)
    grad = -2.0 * (P * rowsum[:, None] - W @ P)
    over = np.maximum(vals - 1.0, 0.0)
    p = mu * 2.0 * over
    Wp = np.zeros((N, N))
    Wp[ii, jj] = p
    Wp[jj, ii] = p
    rowsum_p = Wp.sum(axis=1)
    grad += 2.0 * (P * rowsum_p[:, None] - Wp @ P)
    grad += 2e-3 * P
    return grad


def _cons_fun(x):
    d2 = _pairwise_sq(x.reshape(N, D))
    return 1.0 - d2[_IU]


def _cons_jac(x):
    P = x.reshape(N, D)
    J = np.zeros((N * (N - 1) // 2, N * D))
    ii, jj = _IU
    for k, (a, b) in enumerate(zip(ii, jj)):
        g = 2.0 * (P[a] - P[b])
        J[k, a * D:(a + 1) * D] = -g
        J[k, b * D:(b + 1) * D] = g
    return J


def _fib_sphere(n):
    idx = np.arange(n)
    phi = (1 + np.sqrt(5)) / 2
    z = 1 - (idx + 0.5) / n * 2
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = 2 * np.pi * idx / phi
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    P = np.stack([x, y, z], axis=1)
    d2 = _pairwise_sq(P)
    P = P / np.sqrt(d2.max())
    return P


def _slsqp_polish(P):
    cons = [{'type': 'ineq', 'fun': _cons_fun, 'jac': _cons_jac}]

    def f(x):
        return -_soft_min(_pairwise_sq(x.reshape(N, D))[_IU], 40.0)

    res = minimize(f, P, method='SLSQP', constraints=cons,
                   options={'maxiter': 300, 'ftol': 1e-12})
    return res.x


def construct():
    seeds = [_fib_sphere(N)]
    rng = np.random.default_rng(20240517)
    for _ in range(6):
        P = rng.normal(size=(N, D))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        d2 = _pairwise_sq(P)
        P = P / np.sqrt(d2.max())
        seeds.append(P)

    best_P = seeds[0]
    best_score = _score(best_P)

    for P0 in seeds:
        if _remaining() < 12:
            break
        P = P0
        try:
            res = minimize(_objective_penalty, P0, jac=_grad_penalty,
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-9})
            P = res.x
        except Exception:
            pass
        s = _score(P)
        if s < best_score:
            best_score = s
            best_P = P

        if _remaining() > 30:
            try:
                P2 = _slsqp_polish(P)
                s2 = _score(P2)
                if s2 < best_score:
                    best_score = s2
                    best_P = P2
            except Exception:
                pass

    # final high-precision polish on best
    if _remaining() > 20 and np.isfinite(best_score):
        try:
            P3 = _slsqp_polish(best_P)
            s3 = _score(P3)
            if s3 < best_score:
                best_score = s3
                best_P = P3
        except Exception:
            pass

    if not np.isfinite(_score(best_P)):
        best_P = _fib_sphere(N)
    return best_P
