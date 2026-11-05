"""Rewritten program: minimize value = abs(2*N*max(convolve(h,h))/sum(h)^2)."""
import time
import numpy as np

N = 400
_DEADLINE = 130.0  # safety margin under 150s wall-clock

def _value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

def _value_from_h(h):
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _chirp(p):
    # p = [phase, chirp_rate, amp_cos, amp_sin, base]
    phase, rate, ac, as_, base = p
    i = np.arange(N, dtype=float)
    t = i / N
    freq = rate * t
    ang = phase + 2.0 * np.pi * np.cumsum(freq)  # chirp
    h = base + ac * np.cos(ang) + as_ * np.sin(ang)
    return h

def _two_level(p):
    # p = [a, b, frac, offset]
    a, b, frac, offset = p
    k = int(round(frac * N))
    h = np.empty(N)
    h[:k] = a
    h[k:] = b
    # add a small linear ramp to break symmetry
    h = h + offset * (np.arange(N, dtype=float) / N)
    return h

def _random_h(rng):
    # mix of structured patterns
    kind = rng.integers(0, 4)
    i = np.arange(N, dtype=float)
    if kind == 0:
        f = rng.uniform(1, 20)
        h = np.cos(2 * np.pi * f * i / N + rng.uniform(0, 2*np.pi))
        h = h + rng.uniform(-0.5, 0.5)
    elif kind == 1:
        # two-level
        a = rng.uniform(-2, 2); b = rng.uniform(-2, 2)
        k = rng.integers(1, N)
        h = np.where(np.arange(N) < k, a, b)
    elif kind == 2:
        # chirp
        rate = rng.uniform(0, 50)
        ang = 2 * np.pi * np.cumsum(rate * i / N) + rng.uniform(0, 2*np.pi)
        h = np.cos(ang) + rng.uniform(-0.3, 0.3)
    else:
        # random with smoothing
        h = rng.normal(0, 1, N)
        k = max(1, int(N // 50))
        ker = np.ones(k) / k
        h = np.convolve(h, ker, mode='same')
    # ensure nonzero sum
    s = np.sum(h)
    if abs(s) < 1e-6:
        h = h + 0.1
    return h

def construct():
    start = time.time()
    best_h = np.ones(N)
    best_v = _value_from_h(best_h)

    rng = np.random.default_rng(12345)

    def deadline_ok():
        return (time.time() - start) < _DEADLINE

    # Phase 1: random structured search
    iters = 0
    while deadline_ok() and iters < 4000:
        iters += 1
        h = _random_h(rng)
        v = _value_from_h(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Phase 2: local refinement of best via Nelder-Mead on a low-dim parameterization
    # Try refining via direct coordinate-free: use scipy on a small set of derived params
    # We'll do a few Nelder-Mead runs on chirp params seeded near best if best came from chirp.
    # Simpler: run differential evolution on chirp params (5-dim) with tight budget.
    def chirp_obj(p):
        h = _chirp(p)
        return _value_from_h(h)

    bounds = [(-np.pi, np.pi), (-100, 100), (-2, 2), (-2, 2), (-2, 2)]
    try:
        if deadline_ok():
            res = differential_evolution(
                chirp_obj, bounds,
                maxiter=30, popsize=8, seed=7, tol=1e-4,
                polish=False,
                callback=lambda xk, fk: None,
            )
            if res.x is not None:
                h = _chirp(res.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass

    # Phase 3: local Nelder-Mead on a 2-level parameterization
    def two_obj(p):
        h = _two_level(p)
        return _value_from_h(h)

    bounds2 = [(-3, 3), (-3, 3), (0.01, 0.99), (-1, 1)]
    try:
        if deadline_ok():
            res2 = differential_evolution(
                two_obj, bounds2,
                maxiter=30, popsize=8, seed=13, tol=1e-4, polish=False,
            )
            if res2.x is not None:
                h = _two_level(res2.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass

    # Final: ensure validity and return
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
# EVOLVE-BLOCK-END
