"""Hermite test function for the uncertainty-inequality task.
Maximizes A(g)*A(Fourier(g)) via a high-precision local polish
(Newton on the smooth 3-variable objective) with a global safeguard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_T0 = time.time()
_DEADLINE = 180.0          # hard wall-clock guard (evaluator kills at 210s)
_EPS = 1e-12

# Precomputed constants (exact where possible)
_FAC = [0.0] * 13
_FAC[0] = 1.0
for _k in range(1, 13):
    _FAC[_k] = _FAC[_k - 1] * _k
_FAC = np.array(_FAC)
_SQRT2PI = np.sqrt(2.0 * np.pi)

# M[k] = (2k)! / 2^k  for k = 0..6
_M = np.array([_FAC[2 * k] / (2.0 ** k) for k in range(7)])

# B[k][j] = (2k)! / (2j)! for k >= j else 0  (for g(0) = 0 constraint)
_B = np.zeros((7, 7))
for _k in range(7):
    for _j in range(_k + 1):
        _B[_k, _j] = _FAC[2 * _k] / _FAC[2 * _j]

# g(0) = sum_k a_k * H_{2k}(0) = sum_k a_k * B[k][k] * H_{2k}(0)
# H_{2k}(0) = (-1)^k (2k)! / 2^k = (-1)^k M[k]
_H0 = np.array([(-1.0) ** k * _M[k] for k in range(7)])  # H_{2k}(0)

# a12 = -sum_{k=0}^{5} a_k H_{2k}(0) / H_{12}(0)
_H12_0 = _H0[6]
_A12_COEF = -_H0[:6] / _H12_0  # a12 = A12_COEF . a_free


def _a_full(a_free):
    a12 = float(_A12_COEF @ a_free)
    return np.concatenate([a_free, [a12]])


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _A_g(a_free):
    a = _a_full(a_free)
    return _SQRT2PI * float(np.dot(a, _M))


def _norm_sq(a_free):
    a = _a_full(a_free)
    return float(np.dot(a * a, _M))  # ||g||_2^2 = sum a_k^2 M[k]


def _objective(a_free):
    A = _A_g(a_free)
    if A <= 0.0:
        return -np.inf
    return A * np.sqrt(_norm_sq(a_free))


def _objective_log(a_free):
    A = _A_g(a_free)
    if A <= 0.0:
        return -np.inf
    return np.log(A) + 0.5 * np.log(_norm_sq(a_free))


def _grad_and_hess(a_free):
    a = _a_full(a_free)
    M = _M
    A = _SQRT2PI * float(np.dot(a, M))
    N = float(np.dot(a * a, M))
    if A <= 0.0 or N <= 0.0:
        return None, None
    # f = A * sqrt(N)
    # dA/dfree_k = sqrt(2pi) * (M[k] + A12_COEF[k]*M[6])
    dA = _SQRT2PI * (M[:6] + _A12_COEF * M[6])
    # dN/dfree_k = 2 * a_k * M[k] + 2 * a12 * A12_COEF[k] * M[6]
    a12 = a[6]
    dN = 2.0 * (a[:6] * M[:6] + a12 * _A12_COEF * M[6])
    grad = dA * np.sqrt(N) + A * (dN / (2.0 * np.sqrt(N)))
    # Hessian
    d2N = 2.0 * (M[:6] + _A12_COEF * M[6]) * _A12_COEF * M[6]  # d2N/dfree_k^2
    # Actually d2N/dfree_i dfree_j: from N = sum a_k^2 M_k, a_k linear in free
    # dN/dfree_i = 2 sum_j a_j M_j (da_j/dfree_i)
    # d2N/dfree_i dfree_j = 2 sum_k M_k (da_k/dfree_i)(da_k/dfree_j)
    D = M[:6] + _A12_COEF * M[6]  # da_k/dfree_i pattern: for k<6 it's delta, for k=6 it's A12_COEF
    # Build da/dfree matrix: 7x6
    Da = np.zeros((7, 6))
    Da[:6, :] = np.eye(6)
    Da[6, :] = _A12_COEF
    H2 = 2.0 * (Da * M[:, None]) @ Da.T  # 6x6
    sN = np.sqrt(N)
    H = np.outer(dA, dA) * sN + A * (H2 / (2.0 * sN) - np.outer(dA, dA) * dN @ dN / (4.0 * N * sN))
    return grad, H


def _newton_polish(a0, max_iter=200, ftol=1e-16):
    a = np.asarray(a0, dtype=float).copy()
    f_prev = _objective(a)
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        g, H = _grad_and_hess(a)
        if H is None:
            break
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            step = -g * 0.1
        # Line search (backtracking)
        a_try = a + step
        f_try = _objective(a_try)
        if not np.isfinite(f_try) or f_try < f_prev:
            lam = 0.5
            for _ls in range(60):
                a_try = a + lam * step
                f_try = _objective(a_try)
                if np.isfinite(f_try) and f_try >= f_prev - 1e-16:
                    break
                lam *= 0.5
            else:
                break
        a = a_try
        f = _objective(a)
        if abs(f - f_prev) < ftol * max(1.0, abs(f)):
            break
        f_prev = f
    return a


def _global_search():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)
    # Multistart: log-uniform magnitudes with random signs
    for _ in range(40000):
        if time.time() > _DEADLINE - 20.0:
            break
        mag = 10.0 ** rng.uniform(-6.0, 1.0, size=3)
        sign = rng.choice([-1.0, 1.0], size=3)
        a = sign * mag
        v = _objective(a)
        if np.isfinite(v) and v > best_val:
            best_val = v
            best = a.copy()
    # Grid refine around promising regions
    if best is not None:
        for scale in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
            if time.time() > _DEADLINE - 15.0:
                break
            center = best
            for _ in range(2000):
                if time.time() > _DEADLINE - 15.0:
                    break
                a = center + scale * rng.standard_normal(3)
                v = _objective(a)
                if np.isfinite(v) and v > best_val:
                    best_val = v
                    best = a.copy()
    return best


def construct():
    # Global multistart to find a good basin
    a0 = _global_search()
    if a0 is None:
        a0 = np.array([-1.0, 1e-3, 1e-6])
    # High-precision Newton polish
    a = _newton_polish(a0)
    # Verify it's valid
    val = _objective(a)
    if not np.isfinite(val) or val <= 0:
        a = np.array([-1.0, 1e-3, 1e-6])
    return np.asarray(a, dtype=float)
# EVOLVE-BLOCK-END
