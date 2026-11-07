"""Second autocorrelation inequality: maximize l2/(norm1*norminf) over 50 step heights.

The objective is scale-invariant (only the *shape* of h matters), so we search a
family of parametric profiles for good shapes, then locally refine the 50 heights
with Nelder-Mead and a global L-BFGS-B polish. An explicit wall-clock deadline
guard returns the best result found with a safety margin before the 150 s limit.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0          # safety margin below the 150 s wall-clock kill
_start = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start)


def _value_from_g(g):
    """Fast objective from the autocorrelation g (length 99)."""
    # y = [0, g, 0]  ->  l2 = (w/3) * sum(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Expanding: = (w/3) * (g[0]^2 + 2*sum(g^2) + 2*sum(g[:-1]*g[1:]))
    w = 1.0 / (N + 1)
    g2 = np.dot(g, g)
    gc = np.dot(g[:-1], g[1:])
    l2 = (w / 3.0) * (g2 + 2.0 * g2 + 2.0 * gc)   # g[0]^2 + 2*sum(g^2) = 3*g2
    norm1 = np.sum(np.abs(g)) / (N + 1)
    norminf = np.max(np.abs(g))
    denom = norm1 * norminf
    if denom <= 0.0:
        return 0.0
    return l2 / denom


def _value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    return _value_from_g(g)


def _neg_value(h):
    return -_value(h)


def _neg_shape(params):
    a, b, c = params
    if a <= 1e-9:
        return 1e12
    h = np.exp(-a * (np.arange(N) - (N - 1) / 2.0) ** 2)
    h = h + b * (1.0 + np.cos(2.0 * np.pi * c * np.arange(N) / (N - 1)))
    return -_value(h)


def _gaussian(a, center, width):
    return np.exp(-a * (np.arange(N) - center) ** 2 / (2.0 * width * width))


def _plateau(a, halfw):
    c = (N - 1) / 2.0
    return a * np.exp(-((np.arange(N) - c) / halfw) ** 2)


def _tent(a, halfw):
    c = (N - 1) / 2.0
    return a * np.exp(-np.abs(np.arange(N) - c) / halfw)


def _cosine(a, center, width):
    return a * (0.5 + 0.5 * np.cos(np.pi * (np.arange(N) - center) / width))


def _bimodal(a, sep, width):
    c = (N - 1) / 2.0
    return a * (np.exp(-((np.arange(N) - (c - sep)) ** 2) / (2.0 * width * width)) +
                np.exp(-((np.arange(N) - (c + sep)) ** 2) / (2.0 * width * width)))


def _powerlaw(a, beta):
    c = (N - 1) / 2.0
    return a * np.exp(-beta * np.abs(np.arange(N) - c) ** 0.7)


def _trapezoid(a, inner, outer):
    """Trapezoid shape: flat top of width 2*inner, linear decay to 0 at outer."""
    c = (N - 1) / 2.0
    x = np.abs(np.arange(N) - c)
    h = np.zeros(N)
    mask_inner = x <= inner
    mask_outer = (x > inner) & (x <= outer)
    h[mask_inner] = a
    h[mask_outer] = a * (1.0 - (x[mask_outer] - inner) / (outer - inner))
    return h


def _super_gaussian(a, center, width, p):
    """Super-Gaussian: exp(-(x/w)^p) with p > 2 for sharper peak."""
    x = (np.arange(N) - center) / width
    return a * np.exp(-np.abs(x) ** p)


def _asymmetric_gaussian(a, center, left_w, right_w):
    """Asymmetric Gaussian with different left and right widths."""
    x = np.arange(N) - center
    h = np.zeros(N)
    left_mask = x < 0
    right_mask = x >= 0
    h[left_mask] = a * np.exp(-x[left_mask] ** 2 / (2.0 * left_w * left_w))
    h[right_mask] = a * np.exp(-x[right_mask] ** 2 / (2.0 * right_w * right_w))
    return h


def _refine(h, label):
    """Local Nelder-Mead + global L-BFGS-B polish of a candidate shape."""
    h = np.asarray(h, dtype=float)
    best = h.copy()
    bestv = _value(h)
    if _time_left() < 2.0:
        return best, bestv

    # Nelder-Mead (robust, no gradient)
    try:
        res = minimize(_neg_value, h.copy(), method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-8,
                                'adaptive': True})
        if np.all(np.isfinite(res.x)) and np.all(res.x >= -1e-9):
            cand = np.maximum(res.x, 0.0)
            v = _value(cand)
            if v > bestv:
                bestv, best = v, cand
    except Exception:
        pass
    if _time_left() < 2.0:
        return best, bestv

    # L-BFGS-B global polish (bounded, smooth)
    try:
        res = minimize(_neg_value, best.copy(), method='L-BFGS-B',
                       bounds=[(0.0, None)] * N,
                       options={'maxiter': 500, 'maxls': 30, 'ftol': 1e-12})
        if np.all(np.isfinite(res.x)) and np.all(res.x >= -1e-9):
            cand = np.maximum(res.x, 0.0)
            v = _value(cand)
            if v > bestv:
                bestv, best = v, cand
    except Exception:
        pass
    return best, bestv


def construct():
    global _start
    _start = time.time()

    c = (N - 1) / 2.0
    candidates = []

    # Gaussian family (a, center, width)
    for a in (0.5, 1.0, 2.0, 4.0, 8.0):
        for center in (c, c - 4.0, c + 4.0):
            for width in (1.5, 2.5, 4.0, 6.0, 9.0, 14.0):
                candidates.append(_gaussian(a, center, width))
    # Plateau family
    for a in (1.0, 2.0):
        for halfw in (2.0, 4.0, 6.0, 8.0, 11.0):
            candidates.append(_plateau(a, halfw))
    # Tent family
    for a in (1.0, 2.0):
        for halfw in (2.0, 4.0, 6.0, 9.0, 12.0):
            candidates.append(_tent(a, halfw))
    # Cosine family
    for a in (1.0, 2.0):
        for width in (4.0, 7.0, 10.0, 14.0):
            candidates.append(_cosine(a, c, width))
    # Bimodal family
    for a in (1.0, 2.0):
        for sep in (3.0, 5.0, 7.0):
            for width in (2.0, 3.5, 5.0):
                candidates.append(_bimodal(a, sep, width))
    # Power-law family
    for a in (1.0, 2.0):
        for beta in (0.3, 0.6, 1.0, 1.6):
            candidates.append(_powerlaw(a, beta))
    # Trapezoid family
    for a in (1.0, 2.0):
        for inner in (1.0, 3.0, 5.0):
            for outer in (6.0, 9.0, 12.0, 15.0):
                if outer > inner + 0.5:
                    candidates.append(_trapezoid(a, inner, outer))
    # Super-Gaussian family
    for a in (1.0, 2.0):
        for width in (2.0, 4.0, 6.0, 9.0):
            for p in (3.0, 4.0, 6.0):
                candidates.append(_super_gaussian(a, c, width, p))
    # Asymmetric Gaussian family
    for a in (1.0, 2.0):
        for left_w in (2.0, 4.0, 6.0):
            for right_w in (2.0, 4.0, 6.0):
                if left_w != right_w:
                    candidates.append(_asymmetric_gaussian(a, c, left_w, right_w))
    # Uniform seed (guaranteed valid baseline)
    candidates.append(np.ones(N))

    best = np.ones(N)
    bestv = _value(best)

    # Phase 1: cheap shape-screening over the whole candidate pool
    for h in candidates:
        if _time_left() < 3.0:
            break
        h = np.maximum(h, 0.0)
        v = _value(h)
        if v > bestv:
            bestv, best = v, h

    # Phase 2: refine the top shapes (sorted by screening score)
    scored = []
    for h in candidates:
        h = np.maximum(h, 0.0)
        scored.append((_value(h), h))
    scored.sort(key=lambda t: -t[0])
    for v, h in scored[:15]:
        if _time_left() < 4.0:
            break
        rh, rv = _refine(h, "shape")
        if rv > bestv:
            bestv, best = rv, rh

    # Phase 3: refine the uniform seed too (cheap, safe)
    if _time_left() > 4.0:
        rh, rv = _refine(np.ones(N), "seed")
        if rv > bestv:
            bestv, best = rv, rh

    # Phase 4: second round of refinement on the best result
    if _time_left() > 8.0:
        rh, rv = _refine(best.copy(), "refine2")
        if rv > bestv:
            bestv, best = rv, rh

    # Final safety: ensure a valid, finite, nonnegative array is returned.
    best = np.asarray(best, dtype=float)
    if not (np.all(np.isfinite(best)) and np.all(best >= 0.0) and np.max(best) > 0.0):
        best = np.ones(N)
    return [float(x) for x in best]


if __name__ == "__main__":
    h = construct()
    print("value:", _value(h), "n:", len(h), "min:", min(h))
