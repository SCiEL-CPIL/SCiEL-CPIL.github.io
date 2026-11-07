"""Penalty-based optimizer with sign-change-aware objective for uncertainty inequality."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

_BIG = 1e18

# Even-power coefficients a_k = [coefficient of x^(2k)] for H0, H4, H8, H12.
_H0 = {0: 1.0}
_H4 = {0: 12.0, 1: -48.0, 2: 16.0}
_H8 = {0: 1680.0, 1: -13440.0, 2: 13440.0, 3: -3584.0, 4: 256.0}
_H12 = {
    0: 665280.0, 1: -7983360.0, 2: 13305600.0, 3: -7096320.0,
    4: 1520640.0, 5: -135168.0, 6: 4096.0,
}


def _compute_p_coeffs(c0, c1, c2):
    """Return the coefficients [p5, p4, p3, p2, p1, p0] of p(y)."""
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 665280.0

    def a(k):
        return (c0 * _H0.get(k, 0.0) + c1 * _H4.get(k, 0.0)
                + c2 * _H8.get(k, 0.0) + c3 * _H12.get(k, 0.0))

    return [a(6), a(5), a(4), a(3), a(2), a(1)]


def _value_from(c0, c1, c2):
    """Return r^2/(2*pi) for the given coefficients, or _BIG if ill-defined."""
    coef = _compute_p_coeffs(c0, c1, c2)

    c = list(coef)
    scale = max(1.0, float(np.max(np.abs(c))))
    while len(c) > 1 and abs(c[0]) < 1e-14 * scale:
        c = c[1:]

    try:
        roots = np.roots(c)
    except Exception:
        return _BIG

    real = []
    for r in roots:
        if abs(r.imag) < 1e-7 * max(1.0, abs(r)):
            real.append(float(r.real))
    real.sort()
    if not real:
        return _BIG

    poly = np.poly1d(c)
    cands = []
    n = len(real)
    for i, y0 in enumerate(real):
        if y0 <= 0.0:
            continue
        others = [real[j] for j in range(n) if j != i]
        if others:
            gap = min(abs(o - y0) for o in others)
        else:
            gap = 1.0
        if gap < 1e-9:
            h = 1e-5 * max(1.0, abs(y0))
        else:
            h = 0.25 * gap
        s1 = np.sign(poly(y0 - h))
        s2 = np.sign(poly(y0 + h))
        if s1 != 0 and s2 != 0 and s1 * s2 < 0:
            cands.append(y0)

    if not cands:
        return _BIG
    y_star = max(cands)
    return y_star / (2.0 * math.pi)


def _smooth_value(c0, c1, c2, beta=20.0):
    """Smooth surrogate: penalizes roots below the maximum sign-changing root."""
    coef = _compute_p_coeffs(c0, c1, c2)

    c = list(coef)
    scale = max(1.0, float(np.max(np.abs(c))))
    while len(c) > 1 and abs(c[0]) < 1e-14 * scale:
        c = c[1:]

    try:
        roots = np.roots(c)
    except Exception:
        return _BIG

    real = []
    for r in roots:
        if abs(r.imag) < 1e-7 * max(1.0, abs(r)):
            real.append(float(r.real))
    real.sort()

    if len(real) < 2:
        return _BIG

    poly = np.poly1d(c)

    # Find all sign-changing positive roots
    sign_change_roots = []
    for y0 in real:
        if y0 <= 0.0:
            continue
        h = 1e-4 * max(1.0, abs(y0))
        s1 = poly(y0 - h)
        s2 = poly(y0 + h)
        if s1 * s2 < 0:
            sign_change_roots.append(y0)

    if not sign_change_roots:
        return _BIG

    y_max = max(sign_change_roots)

    # Penalize any sign-changing root that is too large relative to y_max
    # Use a soft-max over all positive sign-changing roots
    penalty = 0.0
    for yr in sign_change_roots:
        penalty += math.exp(beta * (yr - y_max))

    return y_max / (2.0 * math.pi) + 0.1 * penalty / (2.0 * math.pi)


_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def construct():
    start = time.time()
    deadline = start + 170.0

    best_c = _SEED.copy()
    best_v = _value_from(*_SEED)

    def _obj_smooth(c):
        return _smooth_value(c[0], c[1], c[2], beta=15.0)

    def _obj_exact(c):
        return _value_from(c[0], c[1], c[2])

    # Phase 1: Global search with smooth objective
    try:
        de_res = differential_evolution(
            _obj_smooth,
            bounds=[(-30.0, 30.0)] * 3,
            popsize=35,
            maxiter=200,
            tol=1e-8,
            mutation=(0.4, 1.4),
            recombination=0.65,
            seed=12345,
            init="sobol",
            polish=False,
            update_interval=5,
            callback=lambda xk, fk: time.time() > deadline - 120.0,
        )
        if de_res.fun < _BIG:
            de_c = np.asarray(de_res.x, dtype=float)
            de_exact = _obj_exact(de_c)
            if de_exact < best_v:
                best_v = float(de_exact)
                best_c = de_c
    except Exception:
        pass

    # Phase 2: Local polish with exact objective from multiple starting points
    starts = [best_c.copy(), _SEED.copy()]

    # Add perturbed versions of best_c
    rng = np.random.default_rng(99)
    for _ in range(5):
        pert = best_c + rng.normal(0, 0.5, size=3)
        starts.append(pert)

    # Add random starts
    for _ in range(10):
        starts.append(rng.uniform(-30.0, 30.0, size=3))

    for x0 in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            nm = minimize(
                _obj_exact, x0, method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-15,
                         "adaptive": True},
            )
            if nm.fun < best_v:
                best_v = float(nm.fun)
                best_c = np.asarray(nm.x, dtype=float)
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            nm = minimize(
                _obj_exact, best_c, method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-14, "fatol": 1e-16,
                         "adaptive": True},
            )
            if nm.fun < best_v:
                best_v = float(nm.fun)
                best_c = np.asarray(nm.x, dtype=float)
        except Exception:
            pass

    return np.asarray(best_c, dtype=float)
# EVOLVE-BLOCK-END
