"""Structural redesign: analytic double-root elimination + 1D root minimization."""
import numpy as np
import time
from scipy.optimize import minimize_scalar


def _hermite_coeffs(n):
    H0 = [1]
    H1 = [2, 0]
    if n == 0:
        return H0
    if n == 1:
        return H1
    a, b = H0, H1
    for k in range(1, n):
        two_x_b = [2 * c for c in b] + [0]
        sub = [2 * k * c for c in a]
        L = max(len(two_x_b), len(sub))
        two_x_b = [0] * (L - len(two_x_b)) + two_x_b
        sub = [0] * (L - len(sub)) + sub
        res = [two_x_b[i] - sub[i] for i in range(L)]
        while len(res) > 1 and res[0] == 0:
            res = res[1:]
        a, b = b, res
    return b


def _align_add(a, b, scale_b):
    L = max(len(a), len(b))
    a = [0] * (L - len(a)) + a
    b = [0] * (L - len(b)) + b
    return [a[i] + scale_b * b[i] for i in range(L)]


def _poly_eval(coeffs, x):
    v = 0.0
    for c in coeffs:
        v = v * x + c
    return v


def _poly_eval_deriv(coeffs, x):
    n = len(coeffs) - 1
    v = 0.0
    for i in range(n):
        v = v * x + (n - i) * coeffs[i]
    return v


def _g_to_q(gcoeff):
    return gcoeff[2:13]


# Precompute
_H0c = _hermite_coeffs(0)
_H4c = _hermite_coeffs(4)
_H8c = _hermite_coeffs(8)
_H12c = _hermite_coeffs(12)

_H0_0 = _H0c[-1]
_H4_0 = _H4c[-1]
_H8_0 = _H8c[-1]
_H12_0 = _H12c[-1]

_H812 = _align_add(_H8c, _H12c, -_H8_0 / _H12_0)

_L = len(_H12c)
_H0p = [0] * (_L - len(_H0c)) + _H0c
_H4p = [0] * (_L - len(_H4c)) + _H4c
_H812p = [0] * (_L - len(_H812)) + _H812

_qH0 = _g_to_q(_H0p)
_qH4 = _g_to_q(_H4p)
_qH812 = _g_to_q(_H812p)


def _build_q(c0, c1, c2):
    q = [0.0] * len(_qH0)
    for i in range(len(_qH0)):
        q[i] = c0 * _qH0[i] + c1 * _qH4[i] + c2 * _qH812[i]
    return q


def _find_largest_sign_change_root(q):
    try:
        roots = np.roots(q)
    except Exception:
        return None
    real_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-10 * max(1.0, abs(rt.real)):
            real_roots.append(rt.real)
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    for r in real_roots:
        eps = 1e-7 * max(1.0, abs(r))
        left = _poly_eval(q, r - eps)
        right = _poly_eval(q, r + eps)
        if left * right < 0:
            return r
    return None


def _compute_value(c0, c1, c2):
    q = _build_q(c0, c1, c2)
    r = _find_largest_sign_change_root(q)
    if r is None:
        return None
    return r * r / (2.0 * np.pi)


def _valid(c0, c1, c2):
    lead = c0 * _H0p[0] + c1 * _H4p[0] + c2 * _H812p[0]
    if lead <= 0:
        return False
    q = _build_q(c0, c1, c2)
    r = _find_largest_sign_change_root(q)
    if r is None:
        return False
    return True


def _solve_for_double_root(r_target):
    a00 = _poly_eval(_qH0, r_target)
    a01 = _poly_eval(_qH4, r_target)
    b0 = -_poly_eval(_qH812, r_target)
    a10 = _poly_eval_deriv(_qH0, r_target)
    a11 = _poly_eval_deriv(_qH4, r_target)
    b1 = -_poly_eval_deriv(_qH812, r_target)
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-15:
        return None
    c0 = (b0 * a11 - a01 * b1) / det
    c1 = (a00 * b1 - b0 * a10) / det
    c2 = 1.0
    return c0, c1, c2


def _objective(r):
    sol = _solve_for_double_root(r)
    if sol is None:
        return 1e10
    c0, c1, c2 = sol
    if not _valid(c0, c1, c2):
        return 1e10
    val = _compute_value(c0, c1, c2)
    if val is None:
        return 1e10
    return val


def construct():
    seed_c0, seed_c1, seed_c2 = -113 / 100, 1 / 25, 1 / 3240
    seed_val = _compute_value(seed_c0, seed_c1, seed_c2)
    if seed_val is None:
        seed_val = 1.0

    seed_q = _build_q(seed_c0, seed_c1, seed_c2)
    seed_r = _find_largest_sign_change_root(seed_q)
    if seed_r is None:
        seed_r = 1.0

    start_time = time.time()
    best_val = seed_val
    best_coeffs = (seed_c0, seed_c1, seed_c2)

    r_min = max(0.01, seed_r * 0.05)
    r_max = seed_r * 2.0

    deadline = start_time + 150

    # Coarse grid
    n_grid = 300
    r_grid = np.linspace(r_min, r_max, n_grid)
    best_grid_val = seed_val
    best_grid_r = seed_r

    for r in r_grid:
        if time.time() > deadline:
            break
        v = _objective(r)
        if v < best_grid_val:
            best_grid_val = v
            best_grid_r = r

    # Refine
    if best_grid_val < seed_val and best_grid_r > 0.01:
        bracket_width = (r_max - r_min) / max(1, n_grid - 1)
        lo = max(0.005, best_grid_r - 10 * bracket_width)
        hi = best_grid_r + 10 * bracket_width
        try:
            res = minimize_scalar(
                _objective,
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-9, 'maxiter': 300}
            )
            if res.fun < best_grid_val:
                best_grid_val = res.fun
                best_grid_r = res.x
        except Exception:
            pass

    if best_grid_val < seed_val:
        sol = _solve_for_double_root(best_grid_r)
        if sol is not None and _valid(*sol):
            v = _compute_value(*sol)
            if v is not None and v < best_val:
                best_val = v
                best_coeffs = sol

    return np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2]])
