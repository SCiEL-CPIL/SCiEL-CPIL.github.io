"""Erdos minimum overlap - spectral Nesterov optimization."""
import numpy as np
import time

N = 100

# Precompute FFT of the indicator [1,0,0,...,0]
_e = np.zeros(N)
_e[0] = 1.0
_fft_e = np.fft.fft(_e)
_fft_e_conj = np.conj(_fft_e)


def _linear_overlap_values(h):
    """Return C_k = sum_i h[i]*(1-h[(i+k)%N]) for all k via FFT (real)."""
    H = np.fft.fft(h)
    # C_k = N/2 - autocorr_k ; autocorr_k = ifft(|H|^2 * fft_e)[k]
    ac = np.real(np.fft.ifft(np.abs(H) ** 2 * _fft_e_conj))
    return 0.5 * N - ac


def _score(h):
    return float(np.max(_linear_overlap_values(h)))


def _project(v, target):
    """Project onto box [0,1] with sum == target (bisection on shift)."""
    target = float(target)
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = np.clip(v - mid, 0.0, 1.0)
        s = np.sum(c)
        if s > target:
            lo = mid
        else:
            hi = mid
    c = np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)
    s = np.sum(c)
    if s < target:
        idx = np.argsort(c)[::-1][:int(round(target - s + 0.5))]
        c[idx] = np.minimum(c[idx] + (target - s) / max(1, len(idx)), 1.0)
    else:
        idx = np.argsort(c)[:int(round(s - target + 0.5))]
        c[idx] = np.maximum(c[idx] - (s - target) / max(1, len(idx)), 0.0)
    return c


def _smooth_max_grad(vals, beta):
    """Smoothed max of vals with softmax weights; returns (value, gradient wrt vals)."""
    m = np.max(vals)
    x = vals - m
    w = np.exp(beta * x)
    w = w / (np.sum(w) + 1e-300)
    return m + np.log(np.sum(w) + 1e-300) / beta, w


def _grad_overlap(h, beta):
    """Gradient of smoothed-max overlap wrt h (real, length N)."""
    vals = _linear_overlap_values(h)
    _, w = _smooth_max_grad(vals, beta)
    # d C_k / d h_i = (1 - h[(i-k)%N]) ; so grad_h_i = sum_k w_k (1 - h[(i-k)%N])
    # = w_sum - sum_k w_k h[(i-k)%N] = 1 - ifft(FFT(w) * FFT(h))_i ... careful with shift.
    # sum_k w_k h[(i-k)%N] = circular cross-correlation of w and h.
    # = ifft( conj(FFT(w)) * FFT(h) )? Let's verify: (w * h)(i)=sum_k w[k] h[(i-k)%N]
    # FFT of that = FFT(w) * FFT(h) (circular). So the array = ifft(FFT(w)*FFT(h)).
    W = np.fft.fft(w)
    H = np.fft.fft(h)
    conv = np.real(np.fft.ifft(W * H))
    return 1.0 - conv


def _nesterov(h0, max_iter, deadline, beta0=40.0, beta1=80.0):
    """Nesterov accelerated gradient on smoothed-max overlap with projection."""
    h = h0.copy()
    y = h.copy()
    t = 1.0
    best = h.copy()
    best_s = _score(h)
    beta = beta0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        g = _grad_overlap(y, beta)
        # step size: use 2/(L+1) with L ~ 2 (since dC_k/dh bounded by 1, sum over k weighted by 1)
        # Actually grad norm: each component in [0,1]. Use fixed small step.
        step = 2.0 / (it + 3.0)
        h_new = _project(y - step * g, 0.5 * N)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h_new + ((t - 1.0) / t_new) * (h_new - h)
        h = h_new
        t = t_new
        s = _score(h)
        if s < best_s:
            best_s = s
            best = h.copy()
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    # Start from constant 0.5 (known to give N/4)
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_s = _score(h0)

    # Spectral Nesterov with annealed beta
    for beta0, beta1, iters in [(40.0, 80.0, 4000), (80.0, 160.0, 4000), (160.0, 320.0, 3000)]:
        if time.time() > deadline:
            break
        h, s = _nesterov(best_h, iters, deadline, beta0, beta1)
        if s < best_s:
            best_s = s
            best_h = h

    # Final exact projection / cleanup: enforce exact sum
    c = _project(best_h, 0.5 * N)
    s = _score(c)
    if s <= best_s:
        best_h = c
        best_s = s

    # Ensure exact sum and bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = 0.5 * N - np.sum(best_h)
    if diff != 0:
        idx = np.argsort(best_h)[::-1] if diff > 0 else np.argsort(best_h)
        step = diff / N
        best_h = np.clip(best_h + step, 0.0, 1.0)

    return best_h
