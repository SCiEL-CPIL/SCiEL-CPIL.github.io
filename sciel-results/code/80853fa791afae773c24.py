"""Signed step heights minimizing max |autoconv| / integral^2.

Strategy: treat the signed step function as a continuous variable and use
L-BFGS-B to drive down the peak of the autoconvolution. The autoconvolution
is computed with a small FFT (3N instead of 4N) for speed, letting many
optimization iterations fit inside the wall-clock budget. Multiple diverse
starts (sparse, high-frequency, blocky) are tried, then a peak-targeted
polish tightens the dominant lags.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_DEADLINE = time.time() + 140.0  # 150s wall-clock kill -> 10s safety margin


def _autoconv(x):
    """Full linear autoconvolution of x (length 2N-1) via a 3N-size FFT."""
    n = len(x)
    m = 1
    while m < 3 * n:
        m <<= 1
    X = np.fft.rfft(x, m)
    return np.fft.irfft(X * X, m)[:2 * n - 1]


def _score(x):
    """max |autoconv| / (sum x)^2  (lower is better)."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return float(np.max(np.abs(_autoconv(x))) / (s * s))


def _objective(x):
    """Smooth proxy: soft-max of |autoconv| scaled to the score scale."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e6
    ac = _autoconv(x)
    a = np.abs(ac)
    beta = 40.0
    m = a.max()
    # soft-max(a) / s^2  ~=  max|ac| / s^2 when beta large
    sm = m + np.log(np.sum(np.exp(beta * (a - m)))) / beta
    return float(sm / (s * s))


def _peak_polish(x):
    """Target the dominant autoconvolution peaks with a focused L-BFGS run."""
    try:
        res = minimize(_objective, x, method="L-BFGS-B",
                       options={"maxiter": 40, "maxfun": 200, "ftol": 1e-12})
        if res.success or _score(res.x) < _score(x):
            x = res.x
    except Exception:
        pass
    return x


def _run_from(x0, budget):
    """Run L-BFGS-B from x0 within the remaining time budget."""
    x = np.clip(x0, -1.0, 1.0)
    best = x.copy()
    best_s = _score(best)
    try:
        res = minimize(_objective, x, method="L-BFGS-B",
                       options={"maxiter": 60, "maxfun": 400, "ftol": 1e-12,
                                "gtol": 1e-8})
        if res.success:
            x = res.x
        if _score(x) < best_s:
            best = x.copy()
            best_s = _score(best)
    except Exception:
        pass
    if time.time() < _DEADLINE - 10.0:
        best = _peak_polish(best)
    return best, _score(best)


def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 140.0

    rng = np.random.default_rng(12345)
    candidates = []

    # Diverse starts: all-ones, all-zeros-then, sparse, high-freq, blocky, random.
    candidates.append(np.ones(N))
    candidates.append(np.zeros(N))
    candidates.append(np.sign(np.cos(2 * np.pi * np.arange(N) / 8.0)))
    candidates.append(np.sign(np.cos(2 * np.pi * np.arange(N) / 3.0)))
    # blocky alternating
    blk = np.tile(np.array([1.0, 1.0, -1.0, -1.0]), N // 4)[:N]
    candidates.append(blk)
    # sparse few +1, rest 0
    sp = np.zeros(N)
    sp[rng.choice(N, 8, replace=False)] = 1.0
    candidates.append(sp)
    # random signs
    candidates.append(rng.choice([-1.0, 1.0], N))
    # random continuous
    candidates.append(rng.uniform(-1.0, 1.0, N))

    best_x = None
    best_s = np.inf
    for c in candidates:
        if time.time() > _DEADLINE - 15.0:
            break
        x, s = _run_from(c, _DEADLINE)
        if s < best_s:
            best_s = s
            best_x = x

    if best_x is None:
        best_x = np.ones(N)

    # Final tightening if time remains.
    if time.time() < _DEADLINE - 8.0:
        best_x = _peak_polish(best_x)

    return best_x


if __name__ == "__main__":
    x = construct()
    print("score:", _score(x))
