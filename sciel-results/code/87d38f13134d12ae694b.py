"""Erdos minimum overlap: choose step heights in [0,1] summing to N/2
to minimize the maximum overlap of h with shifted (1-h).

Overlap at shift k = sum_i h[i] * (1 - h[i-k]) = N/2 - autocorr(k),
where autocorr(k) = sum_i h[i] h[i-k] (circular).
So minimizing max overlap == maximizing min_k autocorr(k).
"""
import time
import numpy as np

N = 100
DEADLINE = 140.0  # safety margin under 150s kill

def _overlap_profile(h):
    """Return max overlap over all circular shifts of (1-h) against h."""
    # autocorr via FFT: autocorr[k] = sum_i h[i] h[(i-k) mod N]
    H = np.fft.fft(h)
    ac = np.real(np.fft.ifft(H * np.conj(H)))
    # overlap at shift k = N/2 - ac[k]
    return N / 2.0 - ac

def _score(h):
    """Scalar objective: maximum overlap (lower is better)."""
    return float(np.max(_overlap_profile(h)))

def _make_initial():
    """Binary-like balanced sequence (Erdos low-overlap style)."""
    # Start from a random balanced 0/1 sequence, then relax.
    h = np.zeros(N)
    idx = np.random.permutation(N)
    h[idx[: N // 2]] = 1.0
    return h

def _project(h):
    """Project onto [0,1]^N with sum exactly N/2 (simple bisection on offset)."""
    # Clip then adjust by a constant to fix the sum, staying in [0,1].
    lo, hi = 0.0, 1.0
    # We want sum(h - c) = N/2 with h-c in [0,1] -> find c.
    # Equivalent: find t such that sum(clamp(h - t)) = N/2.
    for _ in range(60):
        t = 0.5 * (lo + hi)
        s = np.sum(np.clip(h - t, 0.0, 1.0))
        if s > N / 2.0:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    return np.clip(h - t, 0.0, 1.0)

def construct():
    start = time.time()
    rng = np.random.default_rng(0)

    best = None
    best_val = float("inf")

    def consider(h):
        nonlocal best, best_val
        h = _project(h)
        v = _score(h)
        if v < best_val:
            best_val = v
            best = h.copy()
        return v

    # Phase 1: try several random balanced binary starts and smooth-refine.
    n_starts = 40
    for s in range(n_starts):
        if time.time() - start > DEADLINE:
            break
        h = _make_initial()
        rng.shuffle(h)
        # small random perturbation to break symmetry
        h = h + 0.1 * rng.standard_normal(N)
        consider(h)

    # Phase 2: coordinate / subgradient descent on the smooth objective.
    # Objective f(h) = max_k (N/2 - ac[k]) = N/2 - min_k ac[k].
    # Use a smooth min via log-sum-exp on autocorrelation (maximize min ac)
    # equivalently minimize max overlap with a smooth surrogate.
    def smooth_obj(h, temp):
        prof = _overlap_profile(h)
        # soft-max of overlap profile
        m = np.max(prof)
        return m + temp * np.log(np.sum(np.exp((prof - m) / temp)))

    def smooth_grad(h, temp):
        # derivative of soft-max of overlap w.r.t. h
        prof = _overlap_profile(h)
        m = np.max(prof)
        w = np.exp((prof - m) / temp)
        w /= np.sum(w)
        # d overlap[k]/d h[j]: overlap[k] = N/2 - sum_i h[i]h[i-k]
        # d/dh[j] of -sum_i h[i]h[i-k] = - (h[j-k] + h[j+k])  (circular)
        # so d overlap[k]/d h[j] = -(h[j-k] + h[j+k])
        # grad = sum_k w[k] * (-(h[j-k]+h[j+k]))
        grad = -(np.roll(h, 1) + np.roll(h, -1))  # this is sum over k of 1 * ...
        # Actually each k contributes weight w[k] and the term depends on k:
        # d overlap[k]/dh[j] = -(h[j-k] + h[j+k])
        # grad[j] = -sum_k w[k] (h[j-k] + h[j+k]) = -2 sum_k w[k] h[j-k]
        # because sum_k w[k] h[j+k] = sum_k w[k] h[j-k] (reindex).
        conv = np.real(np.fft.ifft(np.fft.fft(np.roll(h, 1)) * np.fft.fft(w)))
        # careful: sum_k w[k] h[j-k] is circular cross-correlation.
        # Use direct for clarity given N small.
        conv = np.array([np.sum(w * np.roll(h, k)) for k in range(N)])
        grad = -2.0 * conv
        return grad

    h = best.copy()
    temp = 1.0
    lr = 0.02
    it = 0
    while time.time() - start < DEADLINE:
        # subgradient step (descent on smooth_obj)
        g = smooth_grad(h, temp)
        # project gradient step
        h_new = _project(h - lr * g)
        # line search
        base = smooth_obj(h, temp)
        for step in [1.0, 0.5, 0.25, 0.1]:
            cand = _project(h - step * lr * g)
            if smooth_obj(cand, temp) < base:
                h = cand
                break
        else:
            h = h_new
        consider(h)
        # anneal temperature and lr
        temp = max(0.05, temp * 0.99)
        lr = max(0.001, lr * 0.995)
        it += 1
        if it % 50 == 0:
            # occasional random restarts to escape local minima
            if time.time() - start < DEADLINE - 5:
                h2 = best.copy() + 0.2 * rng.standard_normal(N)
                consider(h2)

    # Final projection to guarantee exact sum and bounds.
    if best is None:
        best = np.full(N, 0.5)
    else:
        best = _project(best)

    # Enforce exact sum to machine precision.
    diff = N / 2.0 - np.sum(best)
    # distribute correction to entries not at 0/1 boundary
    free = np.where((best > 1e-9) & (best < 1 - 1e-9))[0]
    if len(free) > 0 and abs(diff) > 1e-12:
        best[free] += diff / len(free)
        best = np.clip(best, 0.0, 1.0)

    return best
