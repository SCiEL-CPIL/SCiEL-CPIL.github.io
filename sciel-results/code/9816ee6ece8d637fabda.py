"""Smooth spectral optimization for Erdos minimum overlap.

overlap(k) = sum_i h[i]*(1-h[(i+k)%N]) = N/2 - autocorr(k)
Minimize max_k overlap(k)  <=>  maximize min_k autocorr(k).
We minimize a smooth log-sum-exp of the overlaps with an exact analytic
gradient, optimized with L-BFGS-B (which handles bounds and the linear
equality constraint sum h = N/2 exactly).
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET = N / 2.0
START = time.time()
DEADLINE = 140.0  # safety margin under the 150 s kill


def _time_left():
    return DEADLINE - (time.time() - START)


def _overlaps(h):
    """overlap(k) = N/2 - autocorr(k) for all shifts k (circular)."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return TARGET - autocorr


def _objective_and_grad(h, beta):
    """Smooth max of overlaps via log-sum-exp and its exact gradient.

    L = m + log(sum(exp(ov - m))) / beta,  m = max(ov)
    dL/dh_j = sum_k exp(ov_k - m) * d(ov_k)/dh_j / (beta * sum exp(ov_k - m))
    d(ov_k)/dh_j = -(h[(j+k)%N] + h[(j-k)%N])
    """
    ov = _overlaps(h)
    m = np.max(ov)
    e = np.exp(ov - m)
    S = np.sum(e)
    L = m + np.log(S) / beta
    # weights w_k = e_k / (beta * S)
    w = e / (beta * S)
    # grad_j = -sum_k w_k (h[(j+k)%N] + h[(j-k)%N])
    # = -(convolve of h with w) - (convolve of h with w reversed)
    conv = np.fft.ifft(np.fft.fft(w) * np.fft.fft(h)).real
    conv_rev = np.fft.ifft(np.fft.fft(w[::-1]) * np.fft.fft(h)).real
    grad = -(conv + conv_rev)
    return L, grad


def _run_lbfgs(h0, beta, budget):
    """Run L-BFGS-B from h0 with the smooth objective; return best feasible h."""
    h0 = np.clip(h0, 0.0, 1.0)
    # ensure sum is close to target before starting
    h0 = h0 + (TARGET - np.sum(h0)) / N
    h0 = np.clip(h0, 0.0, 1.0)

    best = h0.copy()
    best_val = _max_overlap(h0)

    def fun(x):
        v, g = _objective_and_grad(x, beta)
        return v, g

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    t0 = time.time()
    res = minimize(fun, h0, jac=True, method='L-BFGS-B',
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 4000, 'maxfun': 20000,
                            'ftol': 1e-14, 'gtol': 1e-8,
                            'iprint': 0, 'maxls': 50})
    if res.x is not None:
        cand = res.x
        val = _max_overlap(cand)
        if val < best_val:
            best, best_val = cand.copy(), val
    return best, best_val


def _max_overlap(h):
    return float(np.max(_overlaps(h)))


def _feasible(h):
    """Nudge h to satisfy bounds and sum exactly."""
    h = np.clip(h, 0.0, 1.0)
    # fix sum with a few bisection-style adjustments
    for _ in range(8):
        s = np.sum(h)
        if abs(s - TARGET) < 1e-12:
            break
        diff = TARGET - s
        if diff > 0:
            mask = h < 1.0 - 1e-9
        else:
            mask = h > 1e-9
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    # final exact fix on interior points
    s = np.sum(h)
    diff = TARGET - s
    if abs(diff) > 1e-12:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def _make_structured(rng):
    """Diverse structured initializations (banded, low-freq, alternating)."""
    kind = rng.integers(0, 4)
    if kind == 0:
        # banded: some 0s and 1s
        h = np.zeros(N)
        n_ones = int(round(TARGET))
        idx = rng.choice(N, size=n_ones, replace=False)
        h[idx] = 1.0
    elif kind == 1:
        # low-frequency cosine modulation
        a = rng.uniform(0.2, 0.5)
        f = rng.integers(1, 5)
        ph = rng.uniform(0, 2 * np.pi)
        h = 0.5 + a * np.cos(2 * np.pi * f * np.arange(N) / N + ph)
    elif kind == 2:
        # alternating blocks
        bl = int(rng.integers(3, 12))
        h = np.zeros(N)
        for i in range(0, N, bl):
            h[i:i + bl] = rng.uniform(0.0, 1.0)
    else:
        # smooth random
        base = np.random.RandomState(rng.integers(0, 10**6)).rand(N)
        h = 0.5 + 0.45 * (2 * base - 1)
    return _feasible(h)


def construct():
    global START
    START = time.time()
    deadline = time.time() + (DEADLINE - (time.time() - START))

    rng = np.random.default_rng(12345)

    h_best = np.full(N, 0.5)
    best_score = _max_overlap(h_best)

    # Phase 1: L-BFGS-B from structured starts with a fixed smooth beta.
    starts = []
    starts.append(np.full(N, 0.5))
    for _ in range(14):
        starts.append(_make_structured(rng))

    for h0 in starts:
        if time.time() > deadline:
            break
        h, val = _run_lbfgs(h0, beta=8.0, budget=None)
        h = _feasible(h)
        val = _max_overlap(h)
        if val < best_score - 1e-9:
            best_score = val
            h_best = h.copy()

    # Phase 2: anneal beta (sharper max) from the best point.
    for beta in [16.0, 32.0, 64.0, 128.0, 256.0]:
        if time.time() > deadline:
            break
        h, val = _run_lbfgs(h_best, beta=beta, budget=None)
        h = _feasible(h)
        val = _max_overlap(h)
        if val < best_score - 1e-9:
            best_score = val
            h_best = h.copy()

    # Phase 3: coordinate refinement on the true max-overlap objective.
    # For the active shift k*, descent direction is (h[j+k*]+h[j-k*]).
    for _ in range(60):
        if time.time() > deadline:
            break
        ov = _overlaps(h_best)
        k = int(np.argmax(ov))
        improved = False
        for j in range(N):
            a = h_best[(j + k) % N]
            b = h_best[(j - k) % N]
            # candidate: move a toward b (reduce a+b toward 2*0.5=1)
            for (ia, ib) in [((j + k) % N, (j - k) % N)]:
                if ia == ib:
                    continue
                old = (a, b)
                # try setting both to their mean (keeps sum, reduces a+b if a!=b)
                mean = 0.5 * (a + b)
                trial = h_best.copy()
                trial[ia] = mean
                trial[ib] = mean
                trial = _feasible(trial)
                v = _max_overlap(trial)
                if v < best_score - 1e-12:
                    h_best = trial
                    best_score = v
                    improved = True
        if not improved:
            break

    h_best = _feasible(h_best)
    return h_best
