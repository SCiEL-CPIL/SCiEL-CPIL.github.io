import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin under the 150s wall-clock kill

def _fft_autocorr(h):
    """Circular autocorrelation C(k) = sum_i h[i]*h[(i+k) % N] for k=0..N-1."""
    H = np.fft.rfft(h)
    ac = np.fft.irfft(H * np.conj(H), n=N)
    return np.real(ac)

def _autocorr_min(h):
    ac = _fft_autocorr(h)
    return float(np.min(ac[1:]))

def _project_sum_unit(h):
    """Project h onto {h in [0,1]^N : sum h = N/2} (Euclidean projection)."""
    target = N / 2.0
    for _ in range(300):
        s = h.sum()
        if abs(s - target) < 1e-11:
            return h
        lam = (s - target) / N
        h = np.clip(h - lam, 0.0, 1.0)
    return h

def _grad_from_w(h, w):
    """g = sum_{k>=1} w[k-1] * (roll(h,-k) + roll(h,k)), O(N log N) via FFT.
    w has length N-1 (for k=1..N-1)."""
    # build symmetric kernel: K[k]=w[k-1], K[-k]=w[k-1]
    K = np.zeros(N)
    K[1:] = w
    K[:-1] = w  # K[-k] = K[N-k] = w[k-1]
    # g = circular conv of h with K?  (roll(h,-k) = h shifted; sum_k w_k roll(h,-k) = circconv)
    # g = sum_k w_k roll(h,-k) + sum_k w_k roll(h,k)
    # = circconv(h, K) + circconv(h, K) with K symmetric -> 2 * circconv(h,K)
    H = np.fft.rfft(h)
    Kf = np.fft.rfft(K)
    g = np.fft.irfft(H * Kf, n=N)
    g = 2.0 * np.real(g)
    return g

def _feasible_point(t, time_limit):
    """Find h in [0,1]^N, sum=N/2, with C(k) >= t for all k=1..N-1.
    Returns (h, feasible_flag)."""
    h = np.full(N, 0.5)
    best = h.copy()
    best_val = _autocorr_min(h)
    if best_val >= t:
        return best, True
    start = time.time()
    lr = 0.5
    for it in range(40000):
        if time.time() > time_limit:
            break
        ac = _fft_autocorr(h)
        slack = ac[1:] - t
        violated = np.maximum(-slack, 0.0)
        wsum = violated.sum()
        if wsum > 1e-14:
            w = violated / wsum
        else:
            if _autocorr_min(h) >= t:
                return h, True
            w = np.ones(N - 1) / (N - 1)
        g = _grad_from_w(h, w)
        h = h + lr * g
        h = np.clip(h, 0.0, 1.0)
        h = _project_sum_unit(h)
        val = _autocorr_min(h)
        if val > best_val:
            best_val = val
            best = h.copy()
        if best_val >= t:
            return best, True
        lr = max(0.05, lr * 0.999)
    return best, best_val >= t

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    h = np.full(N, 0.5)
    base = _autocorr_min(h)  # = N/4 = 25
    best_h = h.copy()
    best_t = base
    lo = base
    hi = base + 6.0
    # probe to bracket hi (infeasible) above lo (feasible)
    for probe in [base + 1.0, base + 2.0, base + 4.0, base + 6.0]:
        if time.time() > deadline - 25.0:
            break
        hh, ok = _feasible_point(probe, deadline - 20.0)
        if ok:
            lo = probe
            best_t = probe
            best_h = hh
        else:
            hi = probe
            break
    # Bisection between lo (feasible) and hi (infeasible)
    iters = 0
    while (hi - lo) > 0.08 and time.time() < deadline - 12.0 and iters < 10:
        iters += 1
        mid = 0.5 * (lo + hi)
        hh, ok = _feasible_point(mid, deadline - 10.0)
        if ok:
            lo = mid
            best_t = mid
            best_h = hh
        else:
            hi = mid
    # final polish
    if time.time() < deadline - 6.0:
        hh, ok = _feasible_point(best_t + 0.05, deadline - 4.0)
        if ok and _autocorr_min(hh) > _autocorr_min(best_h):
            best_h = hh
    return best_h
