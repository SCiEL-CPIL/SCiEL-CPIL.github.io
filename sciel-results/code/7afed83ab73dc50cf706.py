"""Erdos minimum overlap: minimize max shift-overlap of h with 1-h.

Step heights h on [0,2], 0<=h<=1, integral 1 (sum = N/2).
Overlap at shift s:  sum_i h[i]*(1-h[(i+s)%N])  =  N/2 - (h*h_circ)[s]
=> minimizing max overlap == maximizing min circular autocorrelation.
Autocorrelation computed via FFT (O(N log N)); gradient used by L-BFGS-B.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0
SAFETY = 10.0
_eps = 1e-12


def _autocorr(h):
    """Circular autocorrelation via FFT: r[s] = sum_i h[i] h[(i+s)%N]."""
    f = np.fft.rfft(h, n=N)
    return np.fft.irfft(f * np.conj(f), n=N)[:N]


def _max_overlap(h):
    return float(N / 2.0 - np.min(_autocorr(h)))


def _construct_greedy():
    """Strong baseline: alternating runs of 0 and 1 (sum = N/2)."""
    h = np.zeros(N)
    i = 0
    while i < N:
        h[i] = 1.0
        i += 1
        if i < N:
            h[i] = 0.0
            i += 1
    return h


def construct():
    t0 = time.time()
    best_h = _construct_greedy()
    best_score = _max_overlap(best_h)
    if best_score <= _eps:
        return best_h

    # ---- gradient-based refinement (L-BFGS-B + LogSumExp smoothing) ----
    def _lsmax(x, lam):
        m = np.max(x)
        return m + (1.0 / lam) * np.log(np.sum(np.exp(lam * (x - m))))

    def _objective(h, lam):
        # h in [0,1], sum N/2; minimize max overlap via logsumexp of overlaps
        ov = N / 2.0 - _autocorr(h)
        return float(_lsmax(ov, lam))

    def _grad(h, lam):
        r = _autocorr(h)          # r[s] = sum_i h[i] h[(i+s)%N]
        ov = N / 2.0 - r
        m = np.max(ov)
        p = np.exp(lam * (ov - m))
        p = p / p.sum()
        # d ov[s]/dh[i] = -h[(i-s)%N]  =>  d ov/dh = -(r reversed)
        d_ov = -r[np.mod(np.arange(N) - np.arange(N), N)]
        g = -lam * p @ d_ov
        return g

    def _project(h):
        """Project to {0<=h<=1, sum=N/2} via bisection on water level."""
        target = N / 2.0
        lo, hi = 0.0, 1.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            s = np.sum(np.clip(h - mid, 0.0, 1.0))
            if s > target:
                lo = mid
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        return np.clip(h - mid, 0.0, 1.0)

    # multi-start: greedy + a few random blocky seeds
    seeds = [best_h.copy()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        s = 0
        h = np.zeros(N)
        i = 0
        while i < N:
            v = 1.0 if (rng.random() < 0.5) else 0.0
            h[i] = v
            s += v
            i += 1
            if i < N:
                v = 0.0 if (rng.random() < 0.5) else 1.0
                h[i] = v
                s += v
                i += 1
        seeds.append(h)

    for start in seeds:
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        x0 = start.copy()
        try:
            res = minimize(
                _objective, x0, jac=_grad, args=(40.0,),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 120, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8},
            )
        except Exception:
            continue
        h = _project(res.x)
        sc = _max_overlap(h)
        if sc < best_score - 1e-12:
            best_score = sc
            best_h = h.copy()

    # final feasibility cleanup (sum exactly N/2, values in [0,1])
    h = _project(best_h)
    return h


if __name__ == "__main__":
    h = construct()
    print("sum:", h.sum(), "max_overlap:", _max_overlap(h))
