"""Smooth gradient-based optimization for Erdos minimum overlap."""
import numpy as np
import time


def _max_overlap_exact(h):
    """Compute exact max circular overlap sum_i h[i]*(1-h[(i+s)%N]) over all s."""
    n = len(h)
    f = np.fft.fft(h)
    ac = np.fft.ifft(f * np.conj(f)).real
    return n / 2.0 - np.max(ac)


def _smooth_max_overlap(h, beta):
    """Log-sum-exp surrogate of max overlap; returns (value, grad)."""
    n = len(h)
    f = np.fft.fft(h)
    ac = np.fft.ifft(f * np.conj(f)).real
    o = n / 2.0 - ac  # overlap at each shift
    m = np.max(o)
    # stable softmax
    z = np.exp(beta * (o - m))
    w = z / z.sum()  # softmax weights over shifts
    val = m + np.log(z.sum()) / beta
    # grad: d val/d h[j] = sum_s w_s * d o_s/dh[j]
    # o_s = n/2 - sum_i h[i]*h[(i+s)%n]
    # d o_s/d h[j] = -(h[(j+s)%n] + h[(j-s)%n])   (s != 0), for s=0: -2h[j]
    # So grad_h[j] = - sum_s w_s (h[(j+s)%n]+h[(j-s)%n])   (with s=0 handled)
    # = -( sum_s w_s h[(j+s)%n] + sum_s w_s h[(j-s)%n] )
    # First term = (w conv h)[j], second = (w_rev conv h)[j] where w_rev = w[-s]
    w_rev = np.roll(w, 1)  # w_rev[j] = w[(j-1)%n] corresponds to s -> -s
    g1 = np.fft.ifft(np.fft.fft(w) * np.fft.fft(h)).real
    g2 = np.fft.ifft(np.fft.fft(w_rev) * np.fft.fft(h)).real
    grad_h = -(g1 + g2)
    return val, grad_h


def _project(h, n):
    """Project onto {h in [0,1]^n : sum h = n/2} approximately via bisection on level set."""
    # We want to clip and then adjust sum to n/2.
    # Simple robust approach: find a scalar shift + clip, then fix residual.
    target = n / 2.0
    # Binary search for a threshold shift such that after clipping to [0,1] the sum is target.
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        hh = np.clip(h + mid, 0.0, 1.0)
        s = hh.sum()
        if s > target:
            hi = mid
        else:
            lo = mid
    hh = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Fix any residual exactly on the entry nearest 0.5
    diff = target - hh.sum()
    if abs(diff) > 1e-12:
        idx = np.argmin(np.abs(hh - 0.5))
        hh[idx] += diff
        hh = np.clip(hh, 0.0, 1.0)
    return hh


def construct():
    deadline = time.time() + 110.0
    n = 100
    best_h = np.full(n, 0.5)
    best_score = _max_overlap_exact(best_h)
    rng = np.random.default_rng(12345)

    def try_h(h):
        nonlocal best_h, best_score
        hh = _project(h, n)
        s = _max_overlap_exact(hh)
        if s < best_score - 1e-12:
            best_score = s
            best_h = hh.copy()

    # Gradient descent with log-sum-exp surrogate, multiple beta schedules and restarts
    base = np.full(n, 0.5)
    betas = [20.0, 50.0, 120.0, 300.0, 800.0]
    restarts = 6
    iters_per = 250
    for r in range(restarts):
        if time.time() > deadline:
            break
        h = base.copy()
        if r > 0:
            # random feasible perturbation
            h = np.clip(h + rng.normal(0, 0.15, n), 0.0, 1.0)
            h = _project(h, n)
        # adapt beta across phases
        phase_iters = iters_per // len(betas)
        for beta in betas:
            lr = 0.05 / beta
            for it in range(phase_iters):
                if time.time() > deadline:
                    break
                val, grad = _smooth_max_overlap(h, beta)
                h = h - lr * grad
                h = _project(h, n)
            # periodically accept
            try_h(h)
        try_h(h)

    # Final polish: a few exact-improvement local moves (small random shifts)
    for _ in range(80):
        if time.time() > deadline:
            break
        cand = best_h.copy()
        # random small rotation-like perturbation
        idx = rng.integers(0, n)
        amt = rng.uniform(-0.05, 0.05)
        cand[idx] += amt
        cand[(idx + 1) % n] -= amt
        cand = _project(cand, n)
        try_h(cand)

    # Ensure exact feasibility
    best_h = _project(best_h, n)
    return best_h


# Keep the original name from the seed program
_max_overlap_fft = _max_overlap_exact
