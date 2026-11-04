"""Optimized program: minimize max overlap of h with shifted (1-h)."""
import numpy as np
import time

N = 100


def _overlap_max(h):
    """Max over shifts of (h * (1 - h_shifted)).sum() for circular shifts."""
    h = np.asarray(h, dtype=np.float64)
    best = 0.0
    for s in range(1, N):
        val = np.dot(h, 1.0 - np.roll(h, s))
        if val > best:
            best = val
    return best


def _overlap_max_logsumexp(h, beta=200.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    vals = np.empty(N - 1)
    for i, s in enumerate(range(1, N)):
        vals[i] = np.dot(h, 1.0 - np.roll(h, s))
    m = vals.max()
    if m < -1e12:
        return 0.0
    return m + np.log(np.exp(beta * (vals - m)).sum()) / beta


def _project(h, tol=1e-12, max_iter=200):
    """Project h onto {0<=h<=1, sum(h)=N/2} via bisection on offset."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    # Find offset t such that sum(clamp(h+t)) = target
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        s = np.clip(h + mid, 0.0, 1.0).sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Correct tiny residual
    res = target - h.sum()
    if abs(res) > 1e-15:
        # distribute to non-saturated entries
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if mask.any():
            h[mask] += res / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    t_start = time.time()
    deadline = t_start + 100.0  # safety margin before 120s limit

    # Start from a diverse initial point
    rng = np.random.default_rng(12345)
    h = _project(rng.uniform(0.0, 1.0, N))

    best_h = h.copy()
    best_score = _overlap_max_logsumexp(h, beta=200.0)

    # Gradient descent on the smooth surrogate with projection
    lr = 0.02
    beta = 200.0
    it = 0
    while time.time() < deadline - 5.0:
        # Gradient of sum_s h.(1-roll(h,s)) w.r.t. h
        # d/dh [ h . (1 - roll(h,s)) ] = (1 - roll(h,s)) - roll(1-h, s)
        #   = 1 - roll(h,s) - 1 + roll(h,s) ... wait, need care with roll symmetry
        # Actually: f_s(h) = sum_i h_i (1 - h_{i-s})
        # df_s/dh_i = (1 - h_{i-s}) - h_{i+s}   (circular)
        # So grad = sum_s [(1 - roll(h,s)) - roll(h, -s)]
        grad = np.zeros(N)
        for s in range(1, N):
            grad += (1.0 - np.roll(h, s)) - np.roll(h, -s)
        h_new = _project(h - lr * grad)
        # Adaptive step: accept if surrogate improves
        s_new = _overlap_max_logsumexp(h_new, beta=beta)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
        h = h_new
        it += 1
        # Decay learning rate
        if it % 200 == 0:
            lr = max(0.0005, lr * 0.85)
            beta = min(500.0, beta * 1.05)

    # Final polish: local coordinate ascent / coordinate refinement
    # Try small perturbations on best_h
    h = best_h.copy()
    t_polish = time.time()
    step = 0.01
    while time.time() < deadline - 1.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 1.0:
                break
            for d in (-step, step):
                h2 = h.copy()
                h2[i] = np.clip(h2[i] + d, 0.0, 1.0)
                h2 = _project(h2)
                s2 = _overlap_max(h2)
                s_cur = _overlap_max(h)
                if s2 < s_cur - 1e-12:
                    h = h2
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-5:
                break

    return h
