"""Erdos minimum overlap - FFT-based soft-min optimization + discrete polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET = N / 2.0

# Precompute FFT basis for fast overlap evaluation
_fft_h = np.fft.fft(np.arange(N))


def _overlaps(h):
    """Circular overlaps o_k = sum_j h_j (1 - h_{j+k}) = TARGET - (h*h)_k.
    Uses FFT to compute the circular cross-correlation in O(N log N)."""
    c = np.fft.ifft(np.fft.fft(h) * np.conj(_fft_h)).real
    return TARGET - c


def _max_overlap(h):
    return float(np.max(_overlaps(h)))


def _softmin(ov, t):
    """Smooth minimum of overlaps via log-sum-exp (numerically stable)."""
    m = np.min(ov)
    return m - t * np.log(np.mean(np.exp(-(ov - m) / t)))


def _softmin_grad(ov, t):
    """Gradient of the soft-min w.r.t. h."""
    m = np.min(ov)
    w = np.exp(-(ov - m) / t)
    w /= w.sum()
    # o_k = TARGET - c_k, c_k = sum_j h_j h_{j+k}
    # d o_k / d h_i = - (h_{i+k} + h_{i-k})
    w_ov = np.fft.ifft(np.fft.fft(w) * np.conj(_fft_h)).real
    return -2.0 * w_ov


def _enforce(h):
    """Clip to [0,1] and rescale to exact sum N/2, fixing any residual."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s > 0:
        h = h * (TARGET / s)
        h = np.clip(h, 0.0, 1.0)
    diff = TARGET - h.sum()
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.where(h < 1.0 - 1e-12)[0]
        else:
            idx = np.where(h > 1e-12)[0]
        if idx.size:
            h[idx[0]] += diff
            h = np.clip(h, 0.0, 1.0)
    return h


def _discrete_polish(h0, deadline):
    """Local search over a discrete grid {0, 0.25, 0.5, 0.75, 1}."""
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    h = h0.copy()
    best = _max_overlap(h)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            cur = h[i]
            for g in grid:
                if abs(g - cur) < 1e-9:
                    continue
                cand = h.copy()
                cand[i] = g
                cand = _enforce(cand)
                v = _max_overlap(cand)
                if v < best - 1e-12:
                    best = v
                    h = cand
                    improved = True
                    break
    return h, best


def construct():
    start = time.time()
    deadline = start + 130.0

    rng = np.random.default_rng(12345)

    # ---- Build diverse starting points ----
    starts = []
    starts.append(np.full(N, 0.5))
    # low-frequency sinusoidal patterns (periods 2..8)
    for p in range(2, 9):
        ph = rng.uniform(0, 2 * np.pi)
        starts.append(_enforce(0.5 + 0.4 * np.sin(2 * np.pi * np.arange(N) / p + ph)))
    # random
    for _ in range(6):
        starts.append(_enforce(rng.random(N) * 0.5 + 0.25))
    # bang-bang: half zeros, half ones
    for _ in range(3):
        b = np.zeros(N)
        b[rng.choice(N, N // 2, replace=False)] = 1.0
        starts.append(_enforce(b))
    # alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(_enforce(alt))
    alt2 = np.zeros(N)
    alt2[1::2] = 1.0
    starts.append(_enforce(alt2))

    best_h = None
    best_val = np.inf

    def try_start(h0):
        nonlocal best_h, best_val
        if time.time() > deadline - 3.0:
            return
        h = h0.copy()
        # annealing schedule: start broad, sharpen
        temps = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
        for t in temps:
            if time.time() > deadline - 3.0:
                break

            def obj(x):
                return _softmin(_overlaps(x), t)

            def grad(x):
                return _softmin_grad(_overlaps(x), t)

            try:
                res = minimize(obj, h, jac=grad, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * N,
                               options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8})
                h = _enforce(res.x)
            except Exception:
                h = _enforce(h)
        val = _max_overlap(h)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h.copy()

    for s in starts:
        try_start(s)

    # ---- Discrete polish on the best continuous solution ----
    if best_h is not None and time.time() < deadline - 2.0:
        hp, vp = _discrete_polish(best_h, deadline)
        if vp < best_val - 1e-12:
            best_val = vp
            best_h = hp

    # ---- Final gradient polish with very sharp soft-min ----
    if best_h is not None and time.time() < deadline - 1.0:
        h = best_h.copy()
        for t in [0.01, 0.005]:
            if time.time() > deadline - 1.0:
                break

            def obj(x):
                return _softmin(_overlaps(x), t)

            def grad(x):
                return _softmin_grad(_overlaps(x), t)

            try:
                res = minimize(obj, h, jac=grad, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * N,
                               options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-9})
                h = _enforce(res.x)
            except Exception:
                pass
        val = _max_overlap(h)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _enforce(best_h)
