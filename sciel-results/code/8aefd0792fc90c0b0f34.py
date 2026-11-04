"""Optimized program: smooth log-sum-exp surrogate + gradient descent + high-precision polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
T_END = time.time() + 140.0  # safety margin before the 150s wall-clock kill


def _score_vec(x):
    """Raw objective: (L2^2) / (L1 * sup(|autoconv|))."""
    x = np.asarray(x, dtype=float)
    l2sq = float(np.dot(x, x))
    l1 = float(np.sum(np.abs(x)))
    if l1 <= 0.0:
        return 0.0
    ac = np.convolve(x, x, mode='full')
    s = float(np.max(np.abs(ac)))
    if s <= 0.0:
        return 0.0
    return l2sq / (l1 * s)


def _smooth_objective(x, beta=1e3):
    """Smooth surrogate. Returns (value, gradient) of -ratio with log-sum-exp sup."""
    x = np.asarray(x, dtype=float)
    l1 = float(np.sum(np.abs(x)))
    if l1 <= 0.0:
        return 0.0, np.zeros_like(x)
    ac = np.convolve(x, x, mode='full')
    a = np.abs(ac)
    amax = float(np.max(a))
    if amax <= 0.0:
        return 0.0, np.zeros_like(x)
    u = a / amax
    logsum = float(np.log(np.sum(np.exp(u))))
    logsup = math.log(amax) + logsum
    sup = amax * math.exp(logsum)
    w = np.exp(u - logsum)  # softmax weights summing to 1
    ratio = (float(np.dot(x, x))) / (l1 * sup)
    # d logsup / d x_i = sum_j w_j * d a_j / d x_i
    dlogsup = np.zeros(N)
    for i in range(N):
        # autoconv_j = sum_k x_k x_{j-k}; d/dx_i = sum_k (delta_{j-k,i} x_k + x_k delta_{j,i-k})
        # = x_{j-i} + x_{j-i}  (over valid k), i.e. 2*sum over k with j-k=i or j=i+k
        # autoconv index j in [0, 2N-2]; term x_{j-i} when 0<=j-i<N and x_{j-(N-1-i)}
        # Simpler: d ac_j / d x_i = x_{j-i} + x_{j-(N-1)-i}  (each defined when in [0,N-1])
        row1 = ac_indices = np.arange(0, N)  # j - i
        # j ranges 0..2N-2 ; j-i in [0,N-1] -> j in [i, i+N-1]
        jlo1, jhi1 = i, i + N - 1
        jlo2, jhi2 = (N - 1) - i, (N - 1) - i + N - 1
        # contribution from x_{j-i}: for j in [i, i+N-1], value x_{j-i}
        if jhi1 < 2 * N - 1:
            js1 = np.arange(jlo1, min(jhi1, 2 * N - 2) + 1)
            dlogsup[js1] += w[js1] * x[js1 - i]
        # contribution from x_{j-(N-1)-i}: for j in [N-1-i, 2N-2-i], value x_{j-(N-1)-i}
        jlo2 = max(0, (N - 1) - i)
        jhi2 = min(2 * N - 2, (N - 1) - i + N - 1)
        if jhi2 >= jlo2:
            js2 = np.arange(jlo2, jhi2 + 1)
            dlogsup[js2] += w[js2] * x[js2 - (N - 1) - i]
    # d ratio = d (l2sq/(l1 sup)) = ratio * ( 2 x_i / l2sq - 1/l1 - dlogsup_i )
    l2sq = float(np.dot(x, x))
    if l2sq <= 0.0:
        return 0.0, np.zeros_like(x)
    grad = ratio * (2.0 * x / l2sq - 1.0 / l1 - dlogsup)
    return -ratio, grad


def _normalize(x, target=1.0):
    x = np.asarray(x, dtype=float)
    m = float(np.max(np.abs(x)))
    if m <= 0.0:
        return x
    return x * (target / m)


def _truncated_gauss(center, width):
    x = np.arange(N, dtype=float)
    v = np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    return v


def _smooth_max(x, beta=1e3):
    a = np.asarray(x, dtype=float)
    amax = float(np.max(a))
    return amax + float(np.log(np.sum(np.exp(a - amax)))) / beta


def construct():
    # ---- Phase 1: parametric window sweep (fast, deterministic) ----
    best_x = None
    best_val = 0.0
    for center in np.linspace(0.0, N - 1, 21):
        for width in np.linspace(1.0, N, 21):
            x = _truncated_gauss(center, width)
            x = _normalize(x, 1.0)
            v = _score_vec(x)
            if v > best_val:
                best_val = v
                best_x = x.copy()
    if best_x is None:
        best_x = np.ones(N)

    # ---- Phase 2: smooth gradient-based optimization (log-sum-exp surrogate) ----
    x0 = best_x.copy()
    try:
        res = minimize(
            _smooth_objective,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12},
        )
        if res.x is not None:
            cand = _normalize(res.x, 1.0)
            if _score_vec(cand) > best_val:
                best_val = _score_vec(cand)
                best_x = cand
    except Exception:
        pass

    # ---- Phase 3: high-precision polish (tighter tolerance, more iterations) ----
    try:
        res2 = minimize(
            _smooth_objective,
            best_x,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 8000, 'maxfun': 8000, 'ftol': 1e-16, 'gtol': 1e-14},
        )
        if res2.x is not None:
            cand = _normalize(res2.x, 1.0)
            if _score_vec(cand) > best_val:
                best_val = _score_vec(cand)
                best_x = cand
    except Exception:
        pass

    # ---- Final: scale up for raw score (objective is scale-invariant) ----
    out = [float(v) for v in best_x]
    return out


if __name__ == '__main__':
    r = construct()
    print(len(r), _score_vec(r))
