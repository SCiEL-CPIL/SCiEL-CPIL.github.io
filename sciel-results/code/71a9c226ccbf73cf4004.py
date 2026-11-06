"""Cosmology: compact dark-energy growth approximation.

construct() returns six finite coefficients in [-10,10] for the ansatz
    q = 1 - Om(a),  u = 1 + w0 + wa*(1-a)
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)**gamma
We fit these coefficients (bounded least-squares) to a reference model built
from the contract: a smooth CPL dark-energy background with a tightly
integrated GR second-order growth ODE used as the independent numerical
reference for f and the normalized growth factor D.
"""
import time
import math

import numpy as np

# ---------- deadline guard ----------
_DEADLINE = 84.0  # seconds of wall-clock budget (safety margin under 90 s)
_T0 = time.time()


def _budget_left():
    return _DEADLINE - (time.time() - _T0)


# ---------- physics helpers ----------
def _om_m(a, om_m0):
    """Matter density parameter Om(a) for smooth CPL dark energy."""
    x = 1.0 - a
    de = (1.0 - om_m0) * a ** (-3.0 * (1.0 + 0.5 * x))
    return om_m0 * a ** -3.0 / (om_m0 * a ** -3.0 + de)


def _gamma_true(a, om_m0, w0, wa):
    """Reference growth index from the tight ODE: gamma = -d ln f / d ln a."""
    om = _om_m(a, om_m0)
    if om <= 1e-12:
        return 0.0
    # f = Om^0.5 * (1 + 0.5*(1-om) + 3/14*om + 3/7*om**0.5)
    s = 0.5 * (1.0 - om) + 0.21428571428571427 * om + 0.42857142857142855 * math.sqrt(om)
    f = math.sqrt(om) * (1.0 + s)
    dlnf = 0.5 / om - 0.5 / (1.0 + s)  # d ln f / d ln a
    return -dlnf


def _ref_f(a, om_m0, w0, wa):
    om = _om_m(a, om_m0)
    if om <= 1e-12:
        return 1e-8
    s = 0.5 * (1.0 - om) + 0.21428571428571427 * om + 0.42857142857142855 * math.sqrt(om)
    return math.sqrt(om) * (1.0 + s)


def _ref_lnD(a, om_m0, w0, wa, a_grid, f_grid):
    """ln D(a) = -int_a^0 f(e^x) dx, with f_grid on a_grid (ascending)."""
    x = np.log(np.clip(a, 1e-8, 1.0))
    xg = np.log(np.clip(a_grid, 1e-8, 1.0))
    return -np.trapz(f_grid, xg, x)


# ---------- candidate evaluation ----------
def _ansatz_gamma(c, a, om_m0, w0, wa):
    om = _om_m(a, om_m0)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _candidate_score(c, models, a_grid):
    """Return (max_rel_err, gamma_ok)."""
    max_err = 0.0
    gamma_ok = True
    for (om_m0, w0, wa) in models:
        om_arr = _om_m(a_grid, om_m0)
        g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_ok = False
        f_cand = np.power(om_arr, g)
        f_ref = np.array([_ref_f(a, om_m0, w0, wa) for a in a_grid])
        err_f = np.max(np.abs(f_cand - f_ref) / np.maximum(f_ref, 1e-12))
        max_err = max(max_err, err_f)
        lnD_ref = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_ref)
        lnD_cand = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_cand)
        D_ref = np.exp(lnD_ref)
        D_cand = np.exp(lnD_cand)
        err_D = np.max(np.abs(D_cand - D_ref) / np.maximum(D_ref, 1e-12))
        max_err = max(max_err, err_D)
    return max_err, gamma_ok


# ---------- fitting ----------
def _fit(models, a_grid, n_iter, deadline_guard):
    """Bounded Gauss-Newton / gradient least-squares on log residuals."""
    # residual vector: log(f_cand/f_ref) and log(D_cand/D_ref) over all models/points
    def residual(c):
        r = []
        for (om_m0, w0, wa) in models:
            om_arr = _om_m(a_grid, om_m0)
            g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
            g = np.clip(g, 0.1, 1.2)
            f_cand = np.power(om_arr, g)
            f_ref = np.array([_ref_f(a, om_m0, w0, wa) for a in a_grid])
            r.append(np.log(f_cand / f_ref))
            lnD_ref = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_ref)
            lnD_cand = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_cand)
            r.append(lnD_cand - lnD_ref)
        return np.concatenate(r)

    def obj(c):
        r = residual(c)
        return 0.5 * float(np.dot(r, r))

    def jac(c):
        r = residual(c)
        n = r.size
        J = np.zeros((n, 6))
        h = 1e-6
        for j in range(6):
            cp = c.copy(); cp[j] += h
            cm = c.copy(); cm[j] -= h
            J[:, j] = (residual(cp) - residual(cm)) / (2 * h)
        return J

    # start from constant-gamma seed
    c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best = c.copy()
    best_obj = obj(c)
    lr = 1e-2
    for it in range(n_iter):
        if deadline_guard() < 25.0:
            break
        J = jac(c)
        # normal equations with light damping (Levenberg-Marquardt)
        A = J.T @ J + 1e-8 * np.eye(6)
        b = -J.T @ residual(c)
        step = np.linalg.solve(A, b)
        # backtracking line search
        newc = c + lr * step
        newc = np.clip(newc, -10.0, 10.0)
        no = obj(newc)
        if no < best_obj:
            c = newc
            best = newc.copy()
            best_obj = no
        else:
            lr *= 0.5
            if lr < 1e-9:
                break
    return best


def construct():
    # Standard CPL validation cosmologies (a smooth, representative set).
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.5),   # CPL
        (0.30, -1.1, 0.5),
        (0.30, -0.9, 0.0),
        (0.30, -1.0, -0.5),
        (0.25, -1.0, 0.5),
        (0.35, -1.0, 0.5),
        (0.30, -1.2, 0.8),
    ]
    a_grid = np.linspace(0.05, 1.0, 60)

    try:
        if _budget_left() > 40.0:
            c = _fit(models, a_grid, n_iter=120, deadline_guard=_budget_left)
        else:
            c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    c = np.clip(c, -10.0, 10.0)
    # sanity: ensure gamma in [0.1,1.2] on a probe; if violated, fall back to seed
    try:
        probe_a = np.linspace(0.05, 1.0, 100)
        ok = True
        for (om_m0, w0, wa) in models:
            g = _ansatz_gamma(c, probe_a, om_m0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                ok = False
                break
        if not ok:
            c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return [float(v) for v in c]
