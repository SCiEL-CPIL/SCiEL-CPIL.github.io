import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 143          # explicit wall-clock guard (150s kill -> 7s safety margin)

K = 32
M_STEP = N * K            # 19200
M_FFT = 1 << 15           # 32768

W = 0.5 / N               # width of each interval
RNG = np.random.default_rng(20240517)


def _build_fine(heights):
    """Upsample step heights onto the fine grid (repeat each interval K times)."""
    return np.repeat(heights, K)


def _autoconv(heights):
    """Linear autoconvolution of the upsampled step function (fine grid)."""
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:M_STEP] = _build_fine(heights)
    F = np.fft.rfft(f)
    return np.fft.irfft(F * F, n=M_FFT), f, F


def compute_score(heights):
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    conv, _, _ = _autoconv(heights)
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * W
    if integral_f <= 1e-15:
        return np.inf
    return max_conv / (integral_f ** 2)


def _grad_hardmax(heights):
    """Gradient of max_t (f*f)(t) w.r.t. step heights, via FFT (vectorized)."""
    conv, f, _ = _autoconv(heights)
    p = int(np.argmax(conv))
    # linear cross-correlation C_i = sum_t f_t f_{t+i} == conv[p - i] (circular-safe)
    C = conv[(p - np.arange(M_STEP)) % M_FFT]
    return np.sum(C.reshape(N, K), axis=1)


def _grad_soft(heights, alpha):
    """Smooth (log-sum-exp) surrogate gradient, vectorized via FFT."""
    conv, f, F = _autoconv(heights)
    m = np.max(conv)
    e = np.exp(alpha * (conv - m))
    s = np.sum(e)
    if s <= 0:
        return np.zeros(N)
    w = e / s                                   # softmax weights over fine grid
    # sum_t w_t f_{t+i} == ifft( w_fft * F )[i]
    Wf = np.fft.rfft(w)
    c = np.fft.irfft(Wf * F, n=M_FFT)[:M_STEP]
    return np.sum(c.reshape(N, K), axis=1)


def _score_and_grad(h, alpha):
    h = np.maximum(h, 0.0)
    conv, _, _ = _autoconv(h)
    max_conv = np.max(conv)
    I = np.sum(h) * W
    if I <= 1e-10:
        return 1e12, np.zeros(N)
    score = max_conv / (I ** 2)
    if alpha <= 0:
        g = _grad_hardmax(h)
    else:
        g = _grad_soft(h, alpha)
    dI = np.full(N, W)
    grad = g / (I ** 2) - 2.0 * max_conv * dI / (I ** 3)
    return score, grad


def _objective(h, alpha):
    s, _ = _score_and_grad(h, alpha)
    return s


def _objective_grad(h, alpha):
    s, g = _score_and_grad(h, alpha)
    return s, g


def _structured_candidates():
    """Non-uniform seeds: the non-uniformity is what lets the peak drop below 1.0."""
    cands = []
    x = np.linspace(0.0, 1.0, N)
    t = 2.0 * x - 1.0                            # in [-1, 1]

    # cosine / raised-cosine tapers (plateau with smooth edges)
    for taper_frac in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        h = np.ones(N)
        L = int(taper_frac * N)
        if L > 1:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, L)))
            h[:L] = ramp
            h[-L:] = ramp[::-1]
        cands.append(h)

    # Gaussian bumps (varied width/height)
    for width in (0.25, 0.4, 0.6, 0.8, 1.0):
        for amp in (0.5, 1.0, 1.5, 2.0):
            h = amp * np.exp(-((x - 0.5) / width) ** 2)
            cands.append(h)

    # Parabolic (concave / convex)
    for a in (0.3, 0.6, 1.0, 1.5, 2.0):
        h = 1.0 - a * t * t
        h = np.maximum(h, 0.0)
        if np.max(h) > 0:
            cands.append(h)
        h2 = a * t * t
        cands.append(h2)

    # Linear V / tent
    for slope in (0.5, 1.0, 2.0, 3.0, 5.0):
        h = 1.0 - slope * np.abs(t)
        h = np.maximum(h, 0.0)
        if np.max(h) > 0:
            cands.append(h)
        h2 = slope * np.abs(t)
        cands.append(h2)

    # Two-rectangle (gap in the middle)
    for gap_frac in (0.05, 0.1, 0.2, 0.3, 0.4):
        for height in (0.5, 1.0, 1.5, 2.0):
            h = np.zeros(N)
            gap = int(gap_frac * N)
            left_end = (N - gap) // 2
            right_start = left_end + gap
            if left_end > 0:
                h[:left_end] = height
            if right_start < N:
                h[right_start:] = height
            cands.append(h)

    # Smooth cosine window (Hann-like)
    for a in (0.5, 1.0, 1.5, 2.0):
        h = a * (0.5 * (1.0 + np.cos(np.pi * t)))
        cands.append(h)

    # Random smooth starts
    for _ in range(6):
        r = RNG.normal(1.0, 0.35, N)
        # light smoothing to avoid jagged high-frequency peaks
        ker = np.ones(5) / 5.0
        r = np.convolve(r, ker, mode='same')
        cands.append(np.maximum(r, 0.0))

    return cands


def _optimize_from(h_start, alpha, bounds, maxiter):
    try:
        res = minimize(_objective_grad, h_start, args=(alpha,),
                       method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9})
        return np.maximum(res.x, 0.0), res.fun
    except Exception:
        return h_start, np.inf


def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N

    bounds = [(0.0, None)] * N
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)

    # ---- Phase 1: evaluate structured seeds, keep the best few ----
    scored = []
    for h in _structured_candidates():
        if time.time() - START_TIME > TIME_LIMIT - 25:
            break
        h = np.maximum(h, 0.0)
        if np.sum(h) <= 0:
            continue
        s = compute_score(h)
        scored.append((s, h))
    scored.sort(key=lambda p: p[0])
    for s, h in scored:
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # top distinct starts for optimization
    starts = [best_h.copy()]
    for s, h in scored[:8]:
        if h not in starts:
            starts.append(h.copy())

    # ---- Phase 2: L-BFGS-B with annealed smoothness (hard-max -> soft) ----
    for alpha in (0.0, 1.0, 2.0, 4.0):
        if time.time() - START_TIME > TIME_LIMIT - 8:
            break
        for h_start in starts:
            if time.time() - START_TIME > TIME_LIMIT - 6:
                break
            h_new, s_new = _optimize_from(h_start, alpha, bounds,
                                          maxiter=120 if alpha == 0 else 200)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new.copy()

    # ---- Phase 3: cheap local refinement from the best point ----
    if time.time() - START_TIME < TIME_LIMIT - 4:
        for _ in range(30):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            trial = np.maximum(best_h + RNG.normal(0.0, 0.004, N), 0.0)
            s = compute_score(trial)
            if s < best_score:
                best_score = s
                best_h = trial.copy()

    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]


if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
