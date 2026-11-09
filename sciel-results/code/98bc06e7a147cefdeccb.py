"""Compact dark-energy growth approximation: fit six coefficients to the CPL growth index."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _Om(a, Om0):
    # Matter density at scale factor a for flat CPL universe.
    return Om0 * a ** (-3)

def _target_gamma(a, Om0, w0, wa):
    # Standard growth-index approximation for CPL dark energy.
    Om = _Om(a, Om0)
    u = 1.0 + w0 + wa * (1.0 - a)
    base = Om / u
    if base <= 0:
        return 0.55
    return (5.0 / 7.0) * base ** 0.38

def _design_row(a, Om0, w0, wa):
    Om = _Om(a, Om0)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return np.array([1.0, q, q * q, u, wa * a, u * q])

def _fit_coefficients(deadline):
    # Grid of CPL parameters and scale factors.
    Om0_vals = np.linspace(0.20, 0.35, 12)
    w0_vals = np.linspace(-1.0, -0.6, 12)
    wa_vals = np.linspace(-0.5, 0.5, 12)
    a_vals = np.linspace(0.10, 0.95, 40)

    rows = []
    targets = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    rows.append(_design_row(a, Om0, w0, wa))
                    targets.append(_target_gamma(a, Om0, w0, wa))

    A = np.array(rows)
    b = np.array(targets)

    # Least-squares fit
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    coeffs = np.clip(coeffs, -10.0, 10.0)
    return coeffs

def _gamma_at(a, Om0, w0, wa, c):
    Om = _Om(a, Om0)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

def _objective(c, Om0, w0, wa, a_eval):
    # Relative error of gamma (proxy for f and D) at a few points.
    errs = []
    for a in a_eval:
        Om = _Om(a, Om0)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        g = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q
        gt = _target_gamma(a, Om0, w0, wa)
        errs.append(abs(g - gt) / max(abs(gt), 1e-6))
    return max(errs)

def _polish(c0, deadline):
    # Simple Nelder-Mead-style local polish with time guard.
    from scipy.optimize import minimize
    Om0_vals = np.linspace(0.22, 0.33, 8)
    w0_vals = np.linspace(-1.0, -0.7, 8)
    wa_vals = np.linspace(-0.4, 0.4, 8)
    a_eval = np.linspace(0.15, 0.9, 15)

    # Precompute design rows and targets for speed
    rows = []
    tgts = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_eval:
                    rows.append(_design_row(a, Om0, w0, wa))
                    tgts.append(_target_gamma(a, Om0, w0, wa))
    A = np.array(rows)
    b = np.array(tgts)

    def obj(c):
        pred = A @ c
        err = np.abs(pred - b) / np.maximum(np.abs(b), 1e-6)
        # Max relative error
        return float(np.max(err))

    # Use a smooth surrogate: soft-max via log-sum-exp
    def obj_smooth(c):
        pred = A @ c
        err = np.abs(pred - b) / np.maximum(np.abs(b), 1e-6)
        beta = 200.0
        m = np.max(err)
        return m + math.log(np.sum(np.exp(beta * (err - m)))) / beta

    bounds = [(-10.0, 10.0)] * 6
    try:
        res = minimize(obj_smooth, c0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'maxfun': 5000})
        if time.time() < deadline:
            c1 = np.clip(res.x, -10.0, 10.0)
            if obj(c1) < obj(c0):
                c0 = c1
    except Exception:
        pass

    # Final high-precision polish: exact local solve with tighter tolerance
    if time.time() < deadline:
        try:
            c2, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
            c2 = np.clip(c2, -10.0, 10.0)
            if obj(c2) < obj(c0):
                c0 = c2
        except Exception:
            pass
    return c0

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill

    # Initial least-squares fit
    coeffs = _fit_coefficients(deadline)

    # If time remains, polish
    if time.time() < deadline - 5.0:
        coeffs = _polish(coeffs, deadline)

    # Ensure finite and within bounds
    coeffs = [float(min(max(v, -10.0), 10.0)) for v in coeffs]
    return coeffs
# EVOLVE-BLOCK-END
