"""Model-and-optimize solver for the uncertainty-inequality task."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping

_BIG = 1e18

# Even-power coefficients a_k = [coefficient of x^(2k)] for H0, H4, H8, H12.
# Derived from the recurrence H_{n+1} = 2x H_n - 2n H_{n-1}; constants checked
# against H_{2n}(0) = (-1)^n (2n)!/n!.
_H0 = {0: 1.0}
_H4 = {0: 12.0, 1: -48.0, 2: 16.0}
_H8 = {0: 1680.0, 1: -13440.0, 2: 13440.0, 3: -3584.0, 4: 256.0}
_H12 = {
    0: 665280.0, 1: -7983360.0, 2: 13305600.0, 3: -7096320.0,
    4: 1520640.0, 5: -135168.0, 6: 4096.0,
}

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _compute_p_coeffs(c0, c1, c2):
    """Return the coefficients [p5, p4, p3, p2, p1, p0] of p(y) where y=x^2."""
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 665280.0

    def a(k):
        return (c0 * _H0.get(k, 0.0) + c1 * _H4.get(k, 0.0)
                + c2 * _H8.get(k, 0.0) + c3 * _H12.get(k, 0.0))

    return [a(6), a(5), a(4), a(3), a(2), a(1)]


def _value_from(c0, c1, c2):
    """Return y*/(2*pi) where y* is the largest positive sign-changing root of p(y)."""
    coef = _compute_p_coeffs(c0, c1, c2)

    c = list(coef)
    # Trim leading near-zero coefficients
    scale = max(1.0, float(np.max(np.abs(c))))
    while len(c) > 1 and abs(c[0]) < 1e-14 * scale:
        c = c[1:]

    try:
        roots = np.roots(c)
    except Exception:
        return _BIG

    real = []
    for r in roots:
        if abs(r.imag) < 1e-8 * max(1.0, abs(r)):
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
        # Check sign change
        others = [real[j] for j in range(n) if j != i]
        if others:
            gap = min(abs(o - y0) for o in others)
        else:
            gap = 1.0
        h = min(0.25 * gap, 1e-4 * max(1.0, abs(y0)))
        if h < 1e-12:
            h = 1e-12
        s1 = poly(y0 - h)
        s2 = poly(y0 + h)
        if s1 * s2 < 0:
            cands.append(y0)

    if not cands:
        return _BIG
    y_star = max(cands)
    return y_star / (2.0 * math.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 210s kill, leave 15s safety

    best_c = _SEED.copy()
    best_v = _value_from(*_SEED)

    def _obj(c):
        v = _value_from(c[0], c[1], c[2])
        return v if v < _BIG else _BIG

    # Phase 1: Differential evolution with Sobol initialization
    bounds = [(-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0)]
    
    # Check time
    if time.time() < deadline - 60.0:
        try:
            def _callback(xk, fk):
                return time.time() > deadline - 55.0

            de_res = differential_evolution(
                _obj,
                bounds=bounds,
                popsize=30,
                maxiter=300,
                tol=1e-10,
                mutation=(0.3, 1.3),
                recombination=0.7,
                seed=42,
                init="sobol",
                polish=False,
                callback=_callback,
            )
            if de_res.fun < best_v and de_res.fun < _BIG:
                best_v = float(de_res.fun)
                best_c = np.asarray(de_res.x, dtype=float)
        except Exception:
            pass

    # Phase 2: Basin hopping from best and seed
    def _minimize_wrapper(c):
        """Wrapper for basin hopping's local minimizer."""
        try:
            nm = minimize(
                _obj, c, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-13, "fatol": 1e-15,
                         "adaptive": True},
            )
            return nm.x, nm.fun, False
        except Exception:
            return c, _obj(c), False

    starts = [best_c.copy(), _SEED.copy()]
    rng = np.random.default_rng(123)
    for _ in range(4):
        starts.append(best_c + rng.normal(0, 2.0, size=3))
    for _ in range(6):
        starts.append(rng.uniform(-50.0, 50.0, size=3))

    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            bh = basinhopping(
                _obj, x0, minimizer_kwargs={"args": (), "method": "Nelder-Mead",
                                            "options": {"maxiter": 1500, "xatol": 1e-13,
                                                        "fatol": 1e-15, "adaptive": True}},
                stepsize=2.0,
                niter=8,
                seed=777,
                popsize=5,
                minimizer=_minimize_wrapper,
            )
            if bh.fun < best_v and bh.fun < _BIG:
                best_v = float(bh.fun)
                best_c = np.asarray(bh.x, dtype=float)
        except Exception:
            pass

    # Phase 3: Multiple Nelder-Mead refinements from diverse points
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(999)
        for _ in range(12):
            if time.time() > deadline - 5.0:
                break
            # Mix of perturbations around best and random
            if _ < 6:
                x0 = best_c + rng2.normal(0, 0.5, size=3)
            else:
                x0 = rng2.uniform(-50.0, 50.0, size=3)
            try:
                nm = minimize(
                    _obj, x0, method="Nelder-Mead",
                    options={"maxiter": 5000, "xatol": 1e-14, "fatol": 1e-16,
                             "adaptive": True},
                )
                if nm.fun < best_v and nm.fun < _BIG:
                    best_v = float(nm.fun)
                    best_c = np.asarray(nm.x, dtype=float)
            except Exception:
                pass

    # Phase 4: Final ultra-precise polish
    if time.time() < deadline - 3.0:
        try:
            nm = minimize(
                _obj, best_c, method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-15, "fatol": 1e-17,
                         "adaptive": True},
            )
            if nm.fun < best_v and nm.fun < _BIG:
                best_v = float(nm.fun)
                best_c = np.asarray(nm.x, dtype=float)
        except Exception:
            pass

    return np.asarray(best_c, dtype=float)
# EVOLVE-BLOCK-END
