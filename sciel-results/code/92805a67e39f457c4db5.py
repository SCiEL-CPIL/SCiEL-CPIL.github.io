import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor).
    c[m] = sum_j h[j]*h[j+m] for m=0..N-1
    """
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _autoconv_max(h):
    """Compute max of autoconvolution = w * max_m c_m."""
    c = _autoconv_lags(h)
    return w * c.max()


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    acmax = _autoconv_max(h)
    if acmax <= 0:
        return 0.0
    # ||h||_2^2 = w * s2, ||h||_1 = w * s1, ||h*h||_inf = acmax
    # ratio = (w*s2) / (w*s1 * acmax) = s2 / (s1 * acmax)
    return s2 / (s1 * acmax)


def _ratio_and_grad(h):
    """Compute ratio and analytic gradient (with respect to h, assuming h >= 0).
    Uses scale invariance: we optimize with peak normalized to 1.
    """
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0, np.zeros_like(h)

    c = _autoconv_lags(h)
    m_max = np.argmax(c)
    cmax = c[m_max]
    if cmax <= 0:
        return 0.0, np.zeros_like(h)

    acmax = w * cmax
    ratio = s2 / (s1 * acmax)

    # Gradient of ratio = s2 / (s1 * w * cmax)
    # d(ratio)/d(h_i) = [2*h_i * s1 * cmax - s2 * (1 * cmax + s1 * dcmax/dh_i)] / (s1^2 * w * cmax^2)
    # Simplify: ratio = s2 / (s1 * w * cmax)
    # Let F = s2, G = s1 * cmax (the w factor is constant)
    # ratio = F / (w * G)
    # d(ratio)/dh_i = (dF/dh_i * w*G - F * w * dG/dh_i) / (w*G)^2
    #             = (dF/dh_i * G - F * dG/dh_i) / (w * G^2)

    dF = 2.0 * h  # d(s2)/dh_i

    # dG/dh_i = d(s1*cmax)/dh_i = 1*cmax + s1 * dcmax/dh_i
    # dcmax/dh_i: cmax = c[m_max] = sum_j h[j]*h[j+m_max]
    # dcmax/dh_i = h[i+m_max] (if i+m_max < N) + h[i-m_max] (if i-m_max >= 0)
    # For m_max=0: dcmax/dh_i = 2*h_i
    dcmax = np.zeros_like(h)
    if m_max == 0:
        dcmax[:] = 2.0 * h
    else:
        # c[m] = sum_{j=0}^{N-m-1} h[j]*h[j+m]
        # dc[m]/dh[i] = h[i+m] (if i+m < N, i.e., j=i) + h[i-m] (if i-m >= 0, i.e., j+m=i -> j=i-m)
        dcmax[:N - m_max] += h[m_max:]
        dcmax[m_max:] += h[:N - m_max]

    dG = cmax + s1 * dcmax

    G = s1 * cmax
    grad = (dF * G - s2 * dG) / (w * G * G)

    return ratio, grad


def _objective_neg(x):
    """Negative ratio for minimization (with peak normalization)."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0, np.zeros_like(x)
    h = x / mx
    r, g = _ratio_and_grad(h)
    # Chain rule: x = mx * h, but we're optimizing in h-space effectively
    # Since we normalize, the gradient in x-space is g/mx
    return -r, g / mx


def _objective_neg_scalar(x):
    """Scalar negative ratio."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx
    r, _ = _ratio_and_grad(h)
    return -r


def _initializations():
    """Generate diverse parametric initializations."""
    step_centers = -0.25 + (np.arange(N) + 0.5) * w
    shapes = []

    # Gaussian with various widths
    for sig in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = np.exp(-step_centers**2 / (2 * sig * sig))
        shapes.append(s)

    # Lorentzian / Cauchy
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
        s = 1.0 / (1.0 + (step_centers / sig)**2)
        shapes.append(s)

    # Triangle
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        s = np.clip(1.0 - np.abs(step_centers) / width, 0, None)
        if np.max(s) > 0:
            shapes.append(s)

    # Box / plateau
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        s = (np.abs(step_centers) < width).astype(float)
        shapes.append(s)

    # Single step
    s = np.zeros(N); s[N // 2] = 1.0
    shapes.append(s)

    # Two symmetric steps
    for d in [1, 2, 3, 4, 5, 7, 10]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c + d] = 1
        shapes.append(s)

    # Three symmetric steps
    for d in [2, 3, 5]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c] = 1
            s[c + d] = 1
        shapes.append(s)

    # Power-law decay
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        s = np.exp(-p * np.abs(step_centers))
        shapes.append(s)

    # Sine bump
    for freq in [1, 2, 3, 4, 5, 6, 8]:
        s = np.sin(freq * np.pi * (step_centers + 0.25) / 0.5)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s)

    # Cosine bump (centered)
    for freq in [1, 2, 3, 4]:
        s = np.cos(freq * np.pi * step_centers / 0.25)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s)

    # Smoothstep
    for width in [0.05, 0.1, 0.15, 0.2, 0.25]:
        t = np.clip((width - np.abs(step_centers)) / (2 * width), 0, 1)
        s = t * t * (3 - 2 * t)
        if np.max(s) > 0:
            shapes.append(s)

    # Bump function (compact support, smooth)
    for width in [0.05, 0.08, 0.1, 0.15, 0.2]:
        t = np.abs(step_centers) / width
        s = np.where(t < 1, np.exp(-1.0 / (1.0 - t**2)), 0.0)
        if np.max(s) > 0:
            shapes.append(s)

    # Polynomial (Chebyshev-based)
    t = 2 * step_centers  # map to [-0.5, 0.5]
    for deg in [2, 3, 4, 5, 6, 8, 10, 12]:
        c = np.polynomial.chebyshev.chebval(t, [0] * deg + [1])
        c = np.maximum(c, 0)
        if np.max(c) > 0:
            shapes.append(c / np.max(c))

    # Random smooth shapes
    rng = np.random.default_rng(42)
    for _ in range(15):
        freqs = rng.integers(1, 10, size=10)
        amps = rng.random(10)
        phases = rng.random(10) * 2 * np.pi
        s = np.zeros(N)
        for f, a, ph in zip(freqs, amps, phases):
            s += a * np.cos(f * 2 * np.pi * step_centers + ph)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    # Random with smoothing
    for _ in range(20):
        base = rng.random(N)
        kernel = np.exp(-np.arange(-8, 9)**2 / 8.0)
        kernel /= kernel.sum()
        base = np.convolve(base, kernel, mode='same')
        base = np.maximum(base, 0)
        if np.max(base) > 0:
            shapes.append(base / np.max(base))

    # Pure random
    for _ in range(30):
        shapes.append(rng.random(N))

    return shapes


def _local_optimize(h0, t0, max_iter=2000, tol=1e-12):
    """Run trust-region reflective optimization from h0."""
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    bounds = [(0.0, None)] * N

    try:
        res = minimize(_objective_neg_scalar, h0, method='trust-constr',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'gtol': tol,
                                'xtol': 1e-14, 'barrier_tol': 1e-12,
                                'verbose': 0})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h0, _ratio(h0)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h0)
        if f2 > f1:
            return h2, f2
        return h0, f1
    except Exception:
        pass

    # Fallback: L-BFGS-B
    try:
        res = minimize(_objective_neg_scalar, h0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h0, _ratio(h0)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h0)
        if f2 > f1:
            return h2, f2
        return h0, f1
    except Exception:
        pass

    return h0, _ratio(h0)


def _polish(h, t0):
    """High-precision final polish with tighter tolerances."""
    if _elapsed() > DEADLINE - 3:
        return h, _ratio(h)

    h = np.array(h, dtype=float)
    mx = np.max(h)
    if mx <= 0:
        return h, 0.0
    h = h / mx

    bounds = [(0.0, None)] * N

    # Try SLSQP with tight tolerance
    try:
        res = minimize(_objective_neg_scalar, h, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-15, 'eps': 1e-10})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h, _ratio(h)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass

    # Try L-BFGS-B with very tight tolerance
    try:
        res = minimize(_objective_neg_scalar, h, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h, _ratio(h)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass

    return h, _ratio(h)


def _coordinate_refine(h, t0, max_rounds=20):
    """Coordinate descent refinement."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h)

    for rnd in range(max_rounds):
        if _elapsed() > DEADLINE - 8:
            break
        improved = False
        for i in range(N):
            if _elapsed() > DEADLINE - 8:
                break
            orig = h[i]
            for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01,
                          0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                h[i] = max(0.0, orig + delta)
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    improved = True
                    orig = h[i]
                else:
                    h[i] = orig
        if not improved:
            break

    return h, best_f


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h)

    starts = _initializations()

    # Normalize all starts
    norm_starts = []
    for s in starts:
        s = np.array(s, dtype=float)
        mx = np.max(s)
        if mx > 0:
            norm_starts.append(s / mx)

    bounds = [(0.0, None)] * N

    # Phase 1: Quick screening with fewer iterations
    for idx, s in enumerate(norm_starts):
        if _elapsed() > DEADLINE - 40:
            break
        h, f = _local_optimize(s, t0, max_iter=500, tol=1e-10)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 2: Refine the best with more iterations
    if _elapsed() < DEADLINE - 20:
        best_h, best_f = _local_optimize(best_h, t0, max_iter=2000, tol=1e-13)

    # Phase 3: Coordinate descent refinement
    if _elapsed() < DEADLINE - 10:
        best_h, best_f = _coordinate_refine(best_h, t0, max_rounds=15)

    # Phase 4: High-precision polish
    if _elapsed() < DEADLINE - 4:
        best_h, best_f = _polish(best_h, t0)

    # Phase 5: A few random restarts from perturbed best
    if _elapsed() < DEADLINE - 12:
        rng = np.random.default_rng(999)
        for _ in range(8):
            if _elapsed() > DEADLINE - 8:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0)
            mx = np.max(h_pert)
            if mx <= 0:
                continue
            h_pert = h_pert / mx
            h, f = _local_optimize(h_pert, t0, max_iter=800, tol=1e-12)
            if f > best_f:
                best_f = f
                best_h = h.copy()

    # Final polish if time remains
    if _elapsed() < DEADLINE - 3:
        best_h, best_f = _polish(best_h, t0)

    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
