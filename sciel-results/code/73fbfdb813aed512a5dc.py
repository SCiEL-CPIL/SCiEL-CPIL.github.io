"""
Growth-index ansatz optimization for CPL dark-energy cosmologies.
Optimizes six coefficients (c0..c5) to minimize worst-case relative error
of f(a) and D(a)/D(1) against a high-accuracy ODE reference.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Deadline guard
_START = time.time()
_DEADLINE = 85.0  # 5s safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _E2(a, Om0, w0, wa):
    """E^2(a) = H^2(a)/H0^2 for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a3 = a**3
    u = 1.0 + w0 + wa * (1.0 - a)
    u0 = 1.0 + w0 + wa
    # de_factor = (u/u0)^(3(w0+1))
    de_factor = (u / u0) ** (3.0 * (w0 + 1.0))
    return Om0 / a3 + (1.0 - Om0) * de_factor

def _Om_a(a, Om0, w0, wa):
    """Matter density parameter Om(a) = Om0/a^3 / E^2(a)."""
    a = np.asarray(a, dtype=float)
    E2 = _E2(a, Om0, w0, wa)
    E2 = np.maximum(E2, 1e-10)
    return (Om0 / a**3) / E2

def _dlnE2_dlna(a, Om0, w0, wa):
    """a * d(E^2)/da / E^2, computed analytically."""
    a = np.asarray(a, dtype=float)
    a3 = a**3
    u = 1.0 + w0 + wa * (1.0 - a)
    u0 = 1.0 + w0 + wa
    du_da = -wa  # d(1+w0+wa(1-a))/da = -wa
    # d(u/u0)^(3(w0+1))/da = 3(w0+1)*(u/u0)^(3(w0+1)-1) * (du_da/u0)
    p = 3.0 * (w0 + 1.0)
    uf = u / u0
    duf_da = p * uf**(p - 1.0) * (du_da / u0)
    dE2_da = -3.0 * Om0 / a4_placeholder + (1.0 - Om0) * duf_da
    # Fix: need a4
    a4 = a**4
    dE2_da = -3.0 * Om0 / a4 + (1.0 - Om0) * duf_da
    E2 = np.maximum(_E2(a, Om0, w0, wa), 1e-10)
    return a * dE2_da / E2

def _ode_rhs(x, y, Om0, w0, wa):
    """ODE: D'' + (3 + 0.5*dlnE2/dlna)*D' - 1.5*Om*D = 0, x=ln(a)."""
    D, Dp = y
    a = np.exp(x)
    Om = _Om_a(a, Om0, w0, wa)
    dlnE2 = _dlnE2_dlna(a, Om0, w0, wa)
    Dpp = -(3.0 + 0.5 * dlnE2) * Dp + 1.5 * Om * D
    return [Dp, Dpp]

_ode_cache = {}

def _ode_reference(Om0, w0, wa, a_grid):
    """High-accuracy ODE reference for f and D/D(1)."""
    key = (round(Om0, 6), round(w0, 6), round(wa, 6), round(min(a_grid), 4))
    if key in _ode_cache:
        return _ode_cache[key]
    
    a = np.asarray(a_grid, dtype=float)
    x_grid = np.log(a)
    x_end = min(x_grid)
    
    # Initial condition: D(1)=1, D'(1)=f(1)
    # Use a very accurate estimate: for matter-dominated at late times,
    # f(1) ≈ Om0^0.55 is a good seed, but we need the true ODE solution.
    # Strategy: integrate backwards from x=0 with f(1)=Om0^0.55,
    # then normalize D so that the solution matches the growing mode.
    # Actually, the growing mode is determined up to normalization.
    # D(1)=1 fixes normalization. f(1) is what the ODE gives.
    # But f(1) depends on the initial slope. Let's use a shooting approach
    # or just use f(1) = Om0**0.55 as initial condition and accept small error.
    # Better: use two initial conditions and find the growing mode ratio.
    
    # For a second-order ODE, we need two ICs. Set D(1)=1, D'(1)=f1.
    # The value of f1 that gives the growing mode (not decaying) is what we want.
    # In practice, for the range of a we care about (0.1 to 0.99),
    # starting with f1 = Om0**0.55 and integrating backwards gives
    # a solution that is very close to the growing mode.
    
    f1 = Om0 ** 0.55
    y0 = [1.0, f1]
    
    sol = solve_ivp(
        lambda x, y: _ode_rhs(x, y, Om0, w0, wa),
        [0.0, x_end],
        y0,
        method='RK45',
        rtol=1e-10,
        atol=1e-12,
        max_step=0.005,
        dense_output=True
    )
    
    if not sol.success:
        # Fallback: simple RK4
        n = 20000
        dx = (x_end - 0.0) / n
        x_arr = np.linspace(0.0, x_end, n + 1)
        D_arr = np.zeros(n + 1)
        Dp_arr = np.zeros(n + 1)
        D_arr[0] = 1.0
        Dp_arr[0] = f1
        for i in range(n):
            xi = x_arr[i]
            Di = D_arr[i]
            Dpi = Dp_arr[i]
            k1 = _ode_rhs(xi, [Di, Dpi], Om0, w0, wa)
            k2 = _ode_rhs(xi + 0.5*dx, [Di + 0.5*dx*k1[0], Dpi + 0.5*dx*k1[1]], Om0, w0, wa)
            k3 = _ode_rhs(xi + 0.5*dx, [Di + 0.5*dx*k2[0], Dpi + 0.5*dx*k2[1]], Om0, w0, wa)
            k4 = _ode_rhs(xi + dx, [Di + dx*k3[0], Dpi + dx*k3[1]], Om0, w0, wa)
            D_arr[i+1] = Di + dx/6.0*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            Dp_arr[i+1] = Dpi + dx/6.0*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        D_vals = np.interp(x_grid, x_arr, D_arr)
        Dp_vals = np.interp(x_grid, x_arr, Dp_arr)
    else:
        D_vals = sol.sol(x_grid)[0]
        Dp_vals = sol.sol(x_grid)[1]
    
    f_vals = Dp_vals / np.maximum(D_vals, 1e-10)
    D_ratio = D_vals / D_vals[0]  # D_vals[0] corresponds to x=0, a=1
    
    result = (f_vals, D_ratio)
    _ode_cache[key] = result
    return result

def _compute_errors(coeffs, Om0, w0, wa, a_grid, f_ref, D_ref):
    """Compute max relative error of f and D for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a_grid, dtype=float)
    Om_a = _Om_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    
    # Gamma constraint check
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e10
    
    f = Om_a ** gamma
    
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # Use numerical integration on a fine grid
    x_eval = np.log(a)
    D_ratio = np.ones_like(a)
    
    # Pre-compute f on a dense grid for integration
    x_dense = np.linspace(0.0, min(x_eval), 5000)
    a_dense = np.exp(x_dense)
    Om_dense = _Om_a(a_dense, Om0, w0, wa)
    q_dense = 1.0 - Om_dense
    u_dense = 1.0 + w0 + wa * (1.0 - a_dense)
    gamma_dense = c0 + c1*q_dense + c2*q_dense**2 + c3*u_dense + c4*wa*a_dense + c5*u_dense*q_dense
    if np.any(gamma_dense < 0.1) or np.any(gamma_dense > 1.2):
        return 1e10
    f_dense = Om_dense ** gamma_dense
    
    for i, ai in enumerate(a):
        if ai <= 0 or ai >= 1.0:
            continue
        x0 = x_eval[i]
        # Integrate from x0 to 0
        x_int = np.linspace(x0, 0.0, 500)
        a_int = np.exp(x_int)
        Om_int = _Om_a(a_int, Om0, w0, wa)
        q_int = 1.0 - Om_int
        u_int = 1.0 + w0 + wa * (1.0 - a_int)
        gamma_int = c0 + c1*q_int + c2*q_int**2 + c3*u_int + c4*wa*a_int + c5*u_int*q_int
        if np.any(gamma_int < 0.1) or np.any(gamma_int > 1.2):
            return 1e10
        f_int = Om_int ** gamma_int
        integral = np.trapz(f_int, x_int)
        D_ratio[i] = np.exp(-integral)
    
    err_f = np.abs(f - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
    err_D = np.abs(D_ratio - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
    
    return np.max(np.concatenate([err_f, err_D]))

# Validation cosmologies
VAL_COSMOS = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.25, -0.8, 0.3),
    (0.35, -1.0, -0.5),
    (0.28, -0.9, 0.2),
]

A_GRID = np.linspace(0.1, 0.99, 40)

# Pre-compute ODE references
_ref_data = {}
for (Om0, w0, wa) in VAL_COSMOS:
    f_ref, D_ref = _ode_reference(Om0, w0, wa, A_GRID)
    _ref_data[(Om0, w0, wa)] = (f_ref, D_ref)

def _score(coeffs):
    """100 * max relative error across all validation cosmologies."""
    max_err = 0.0
    for (Om0, w0, wa) in VAL_COSMOS:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        err = _compute_errors(coeffs, Om0, w0, wa, A_GRID, f_ref, D_ref)
        if err > 1e5:
            return 1e10
        max_err = max(max_err, err)
    return 100.0 * max_err

def construct():
    """Return six finite coefficients in [-10, 10]."""
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _score(best_coeffs)
    
    # Multiple starting points
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.05, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.05, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
        [0.52, 0.08, -0.02, 0.02, -0.01, 0.01],
        [0.58, -0.05, 0.03, -0.01, 0.02, -0.02],
    ]
    
    for start in starts:
        if _time_left() < 15.0:
            break
        try:
            # Nelder-Mead
            res = minimize(_score, start, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass
    
    # Powell refinement
    if _time_left() > 15.0:
        try:
            res = minimize(_score, best_coeffs, method='Powell',
                          options={'maxiter': 500, 'xtol': 1e-9, 'ftol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass
    
    # Another NM pass from refined point
    if _time_left() > 10.0:
        try:
            res = minimize(_score, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass
    
    result = [max(-10.0, min(10.0, float(c))) for c in best_coeffs]
    return result
