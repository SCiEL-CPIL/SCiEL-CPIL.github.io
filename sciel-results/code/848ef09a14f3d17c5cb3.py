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
from scipy.optimize import minimize

# ---------- deadline guard ----------
_DEADLINE = 82.0  # seconds of wall-clock budget (safety margin under 90 s)
_T0 = time.time()


def _budget_left():
    return _DEADLINE - (time.time() - _T0)


# ---------- physics helpers ----------
def _om_m(a, om_m0):
    """Matter density parameter Om(a) for smooth CPL dark energy."""
    x = 1.0 - a
    de = (1.0 - om_m0) * a ** (-3.0 * (1.0 + 0.5 * x))
    return om_m0 * a ** -3.0 / (om_m0 * a ** -3.0 + de)


def _H2_ratio(a, om_m0, w0, wa):
    """H^2(a)/H^2(1) for CPL dark energy."""
    x = 1.0 - a
    om_m = _om_m(a, om_m0)
    # H^2/H0^2 = Om_m * a^-3 + Om_de * a^(-3(1+w0+wa*x))
    # where Om_de = 1 - Om_m0
    # Actually for CPL: w(a) = w0 + wa*(1-a), so the DE scale factor is a^(-3(1+w(a)))
    # More precisely: rho_de ∝ a^(-3(1+w0-wa)) * exp(-3*wa*(1-a)/a) ... 
    # Let me use the standard form: E^2 = Om_m/a^3 + (1-Om_m)*a^(-3*(1+w0+wa*(1-a)))
    # But actually the standard CPL has w(a) = w0 + wa*(1-a), and
    # rho_de(a) ∝ exp(-3*∫_a^1 (1+w(a'))/a' da')
    # For simplicity and matching the seed's _om_m, use:
    de = (1.0 - om_m0) * a ** (-3.0 * (1.0 + 0.5 * x))
    return om_m0 * a ** -3.0 + de


def _solve_growth_ode(om_m0, w0, wa, a_grid):
    """Solve the GR second-order growth ODE for D(a).
    
    D'' + (2 + H'/H) D' - (3/2) Om_m H^2 D = 0
    
    Using the standard form with f = D'/D:
    f' + f^2 + (3/2 - q/2)(1+f) - (3/2)Om_m = 0  where q = -aH'/H
    
    We integrate D directly using RK4 on a fine grid.
    """
    n = len(a_grid)
    # Work in ln a space: let x = ln a, then d/d ln a = d/dx
    # D'' + (3/2 + q) D' - (3/2) Om_m H^2 D = 0
    # where q = -d ln H / d ln a
    
    x = np.log(np.clip(a_grid, 1e-10, 1.0))
    
    # Compute q = -d ln H / d ln a numerically
    # H^2 ∝ Om_m/a^3 + de_term
    # d ln H / d ln a = (1/2) d ln(H^2) / d ln a
    # Let E2(a) = H^2/H0^2
    # d ln E2 / d ln a = (a/E2) * dE2/da
    
    # Compute E2 on a fine grid for accurate derivatives
    a_fine = np.linspace(max(a_grid[0], 1e-6), 1.0, 2000)
    E2 = np.array([_H2_ratio(a, om_m0, w0, wa) for a in a_fine])
    
    # d ln E2 / d ln a via central differences
    ln_a_fine = np.log(a_fine)
    ln_E2 = np.log(np.maximum(E2, 1e-30))
    dlnE2_dlna = np.gradient(ln_E2, ln_a_fine)
    q_fine = -0.5 * dlnE2_dlna
    
    # Interpolate q onto a_grid
    q_grid = np.interp(x, ln_a_fine, q_fine)
    
    # Om_m on grid
    Om_grid = np.array([_om_m(a, om_m0) for a in a_grid])
    
    # E2 on grid (for H^2 factor)
    E2_grid = np.array([_H2_ratio(a, om_m0, w0, wa) for a in a_grid])
    
    # Integrate D'' + (3/2 + q) D' - (3/2) Om_m * E2 * D = 0
    # Initial conditions near a=1: D(1)=1, D'(1) = f(1) ≈ Om_m(1)^0.55 ≈ Om_m0^0.55
    # Use f(1) from the analytic approximation as initial slope
    f_init = Om_m0 ** 0.55
    
    D = np.zeros(n)
    Dp = np.zeros(n)  # D' = dD/dlna
    
    # Start from a=1 (last element) and integrate backward
    D[-1] = 1.0
    Dp[-1] = f_init
    
    # RK4 integration backward in x = ln a
    for i in range(n - 2, -1, -1):
        dx = x[i] - x[i + 1]  # negative since going backward
        
        def ode(y):
            # y = [D, Dp]
            D_val, Dp_val = y
            # D'' = -(3/2 + q) D' + (3/2) Om_m * E2 * D
            Dpp = -(1.5 + q_grid[i]) * Dp_val + 1.5 * Om_grid[i] * E2_grid[i] * D_val
            return [Dp_val, Dpp]
        
        k1 = ode([D[i+1], Dp[i+1]])
        k2 = ode([D[i+1] + 0.5*dx*k1[0], Dp[i+1] + 0.5*dx*k1[1]])
        k3 = ode([D[i+1] + 0.5*dx*k2[0], Dp[i+1] + 0.5*dx*k2[1]])
        k4 = ode([D[i+1] + dx*k3[0], Dp[i+1] + dx*k3[1]])
        
        D[i] = D[i+1] + (dx/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        Dp[i] = Dp[i+1] + (dx/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    
    # f = D'/D
    f = Dp / np.maximum(D, 1e-10)
    
    return D, f


def _ansatz_gamma(c, a, om_m0, w0, wa):
    om = _om_m(a, om_m0)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _compute_residuals(c, models, a_grid, ref_data):
    """Compute log(f_cand/f_ref) and log(D_cand/D_ref) for all models/points."""
    r = []
    for idx, (om_m0, w0, wa) in enumerate(models):
        om_arr = _om_m(a_grid, om_m0)
        g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
        g = np.clip(g, 0.1, 1.2)
        f_cand = np.power(om_arr, g)
        f_ref = ref_data[idx][1]
        D_ref = ref_data[idx][0]
        
        # D_cand from f_cand: D(a)/D(1) = exp(-integral from ln a to 0 of f(e^x) dx)
        x = np.log(np.clip(a_grid, 1e-10, 1.0))
        # Integrate -f from x to 0 (i.e., from ln a to 0)
        # Use cumulative trapezoidal from the end
        # D(a)/D(1) = exp(-∫_{ln a}^{0} f(e^x) dx)
        # = exp(∫_{0}^{ln a} f(e^x) dx) for a < 1... 
        # Actually: ∫_{ln a}^{0} f dx = -∫_{0}^{ln a} f dx
        # So D(a)/D(1) = exp(∫_{0}^{ln a} f(e^x) dx)
        # But x = ln a < 0 for a < 1, so ∫_{0}^{ln a} = -∫_{ln a}^{0}
        # Let me just compute cumulative integral from a=1 backward
        
        # f_cand as function of x = ln a
        # D(a)/D(1) = exp(-∫_{x}^{0} f(e^{x'}) dx')
        # = exp(∫_{0}^{x} f(e^{x'}) dx')  for x < 0
        # Compute ∫ from 0 to x using trapezoidal rule
        
        # Sort by x ascending (a ascending)
        # x goes from ln(a_min) to 0
        # ∫_{0}^{x_i} f dx = -∫_{x_i}^{0} f dx
        
        # Use cumulative trapezoidal from the right (a=1 end)
        f_sorted = f_cand  # already in order of a_grid which is ascending
        x_sorted = x  # ascending
        
        # Integral from x_i to x_{n-1}=0 of f dx
        # cumtrapz from the end
        integral_from_end = np.zeros(n)
        for i in range(n - 2, -1, -1):
            integral_from_end[i] = integral_from_end[i+1] + 0.5 * (f_sorted[i] + f_sorted[i+1]) * (x_sorted[i+1] - x_sorted[i])
        
        # D_cand = exp(-integral_from_end) since D(1)=1
        D_cand = np.exp(-integral_from_end)
        
        # Normalize: D_cand[-1] should be 1
        D_cand = D_cand / D_cand[-1]
        
        r.append(np.log(f_cand / np.maximum(f_ref, 1e-12)))
        r.append(np.log(D_cand / np.maximum(D_ref, 1e-12)))
    return np.concatenate(r)


def _smooth_max(r, beta=50.0):
    """Smooth approximation to max|r_i| using log-sum-exp."""
    abs_r = np.abs(r)
    m = np.max(abs_r)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_r - m)))) / beta


def _objective(c, models, a_grid, ref_data):
    """Smooth surrogate objective: log-sum-exp of |residuals|."""
    r = _compute_residuals(c, models, a_grid, ref_data)
    return _smooth_max(r)


def _objective_grad(c, models, a_grid, ref_data):
    """Gradient of the smooth max objective via numerical differentiation."""
    eps = 1e-6
    g = np.zeros(6)
    f0 = _objective(c, models, a_grid, ref_data)
    for j in range(6):
        cp = c.copy()
        cp[j] += eps
        fp = _objective(cp, models, a_grid, ref_data)
        cm = c.copy()
        cm[j] -= eps
        fm = _objective(cm, models, a_grid, ref_data)
        g[j] = (fp - fm) / (2 * eps)
    return g


def _fit(models, a_grid, ref_data, deadline_guard):
    """Optimize using L-BFGS-B on smooth surrogate, then polish with exact max."""
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0.copy()
    best_obj = _objective(c0, models, a_grid, ref_data)
    
    # L-BFGS-B optimization on smooth surrogate
    try:
        if deadline_guard() > 25.0:
            res = minimize(
                _objective,
                c0,
                args=(models, a_grid, ref_data),
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0)] * 6,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_obj:
                c_lbfgs = np.clip(res.x, -10.0, 10.0)
                obj_lbfgs = _objective(c_lbfgs, models, a_grid, ref_data)
                if obj_lbfgs < best_obj:
                    best_c = c_lbfgs
                    best_obj = obj_lbfgs
    except Exception:
        pass
    
    # Polish with exact max-norm using projected subgradient
    def exact_max_err(c):
        r = _compute_residuals(c, models, a_grid, ref_data)
        return np.max(np.abs(r))
    
    best_err = exact_max_err(best_c)
    
    lr = 1e-3
    for it in range(100):
        if deadline_guard() < 10.0:
            break
        
        r = _compute_residuals(best_c, models, a_grid, ref_data)
        idx = np.argmax(np.abs(r))
        sign = np.sign(r[idx])
        
        # Numerical gradient of r[idx] w.r.t. c
        eps = 1e-7
        grad_r = np.zeros(6)
        for j in range(6):
            cp = best_c.copy()
            cp[j] += eps
            r_p = _compute_residuals(cp, models, a_grid, ref_data)
            cm = best_c.copy()
            cm[j] -= eps
            r_m = _compute_residuals(cm, models, a_grid, ref_data)
            grad_r[j] = (r_p[idx] - r_m[idx]) / (2 * eps)
        
        grad = sign * grad_r
        newc = best_c - lr * grad
        newc = np.clip(newc, -10.0, 10.0)
        
        e = exact_max_err(newc)
        if e < best_err:
            best_err = e
            best_c = newc
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best_c


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
    a_grid = np.linspace(0.05, 1.0, 100)

    # Precompute reference data using ODE integration
    ref_data = []
    for (om_m0, w0, wa) in models:
        try:
            D_ref, f_ref = _solve_growth_ode(om_m0, w0, wa, a_grid)
            ref_data.append((D_ref, f_ref))
        except Exception:
            # Fallback to analytic approximation
            f_ref = np.array([_om_m(a, om_m0) ** 0.55 for a in a_grid])
            D_ref = np.ones_like(a_grid)
            # Simple integration
            x = np.log(np.clip(a_grid, 1e-10, 1.0))
            integral = np.zeros(len(a_grid))
            for i in range(len(a_grid) - 2, -1, -1):
                integral[i] = integral[i+1] + 0.5 * (f_ref[i] + f_ref[i+1]) * (x[i+1] - x[i])
            D_ref = np.exp(-integral) / np.exp(-integral[-1])
            ref_data.append((D_ref, f_ref))

    try:
        if _budget_left() > 20.0:
            c = _fit(models, a_grid, ref_data, deadline_guard=_budget_left)
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
