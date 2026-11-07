import time
import numpy as np

N = 50
_START = time.time()
_DEADLINE = 110.0  # safety margin under 120s limit


def _value(g):
    """Compute l2/(norm_1*norm_inf) from autocorrelation g (nonnegative)."""
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum_i (w/3)(y_i^2 + y_i y_{i+1} + y_{i+1}^2)
    a = y[:-1]
    b = y[1:]
    l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _score(h):
    g = np.convolve(h, h)
    return _value(g)


def _refine(h, budget=0.5):
    """Local coordinate ascent on peak-normalized ratio. Returns improved h."""
    best = h.copy()
    best_s = _score(best)
    n = len(best)
    t_end = time.time() + budget
    # random coordinate order
    order = np.random.permutation(n)
    it = 0
    while time.time() < t_end:
        it += 1
        progress = False
        for idx in order:
            if time.time() > t_end:
                break
            base = best[idx]
            # candidate multipliers around 1 (geometric)
            for f in (0.5, 0.75, 1.25, 1.5, 2.0, 0.0):
                trial = best.copy()
                trial[idx] = base * f
                s = _score(trial)
                if s > best_s + 1e-12:
                    best = trial
                    best_s = s
                    progress = True
                    break
        if not progress:
            break
    return best, best_s


def construct():
    """Return 50 nonnegative step heights maximizing l2/(norm_1*norm_inf)."""
    global _START, _DEADLINE
    if _START is None:
        _START = time.time()
        _DEADLINE = _START + 110.0

    rng = np.random.default_rng(12345)

    # Strong prior: smooth peaked bump (normalized peak of g to 1)
    x = np.linspace(-1.0, 1.0, N)
    bump = np.exp(-4.0 * x * x)
    bump = np.maximum(bump, 1e-6)
    h0 = bump / np.max(np.convolve(bump, bump))  # peak of g = 1

    best = h0.copy()
    best_s = _score(best)

    # Local refinement of the bump
    h_r, s_r = _refine(h0, budget=2.0)
    if s_r > best_s:
        best, best_s = h_r, s_r

    # Random restarts within budget (bump + noise, then refine)
    while time.time() < _DEADLINE - 4.0:
        sigma = rng.uniform(0.15, 0.7)
        cand = np.maximum(h0 + sigma * rng.standard_normal(N), 0.0)
        if np.max(cand) <= 0:
            continue
        cand = cand / np.max(np.convolve(cand, cand))
        # quick refinement
        h_r, s_r = _refine(cand, budget=0.25)
        if s_r > best_s + 1e-12:
            best, best_s = h_r, s_r

    # Final safety: ensure finite, nonnegative
    best = np.asarray(best, dtype=float)
    best[~np.isfinite(best)] = 0.0
    best = np.maximum(best, 0.0)

    return list(best)
