"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_TIME_BUDGET = 75.0  # seconds, leaving 15s margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a for flat CPL DE."""
    a = np.asarray(a, dtype=float)
    Om_m = Om0 * a**(-3)
    Om_de = (1.0 - Om0) * a**(-3.0 * (1.0 + w0 + wa * (1.0 - a)))
    return Om_m / (Om_m + Om_de)

def _gamma_of_a(a, c, Om0, w0, wa):
    """Compute gamma from the six-coefficient ansatz."""
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q)

def _f_approx(a, c, Om0, w0, wa):
    """Approximate growth rate f = Om(a)^gamma(a)."""
    Om = _Om_of_a(a, Om0, w0, wa)
    g = _gamma_of_a(a, c, Om0, w0, wa)
    return Om**g

def _f_ref(a, Om0, w0, wa):
    """Reference growth rate from the growth ODE (Einstein equation)."""
    # f = d ln D / d ln a satisfies:
    # f' + (3/2 + H'/H) f + (3/2) Om = 0  (in terms of ln a)
    # At a=1: f(1) = Om(1)^gamma ~ Om0^0.55 as initial condition
    # But we need the true ODE solution. Use the standard relation:
    # f = d ln D / d ln a, and the ODE is:
    # f'' + (3 + H'/H) f' + (3/2) Om(a) f = 0
    # Actually the standard growth ODE in terms of f is:
    # f' + (3/2 + H'/H) f = - (3/2) Om(a)  ... no
    # Let me use the standard form:
    # d^2 D/d ln a^2 + (3/2 + H'/H) dD/d ln a - (3/2) Om(a) D = 0
    # With f = dD/d ln a:
    # df/d ln a + (3/2 + H'/H) f - (3/2) Om(a) = 0
    # So: f' = (3/2) Om(a) - (3/2 + H'/H) f
    # where prime is d/d ln a
    # H'/H = d ln H / d ln a = (dH/da) / (H) * a = a H' / H
    # For CPL: H^2/H0^2 = Om0 a^-3 + Omd0 a^{-3(1+w0+wa)}
    # d ln H / d ln a = [ -3 Om0 a^-3 + -3(1+w0+wa) Omd0 a^{-3(1+w0+wa)} ] / [2 H^2/H0^2]
    # = [ -3 Om_m - 3(1+w0+wa) Om_de ] / [2 (Om_m + Om_de)]
    # = -3/2 [ Om_m + (1+w0+wa) Om_de ] / [ Om_m + Om_de ]
    # = -3/2 [ 1 + w0+wa - w0*Om_m - wa*Om_m ... let me just compute numerically
    
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    Om_de = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    # d ln H / d ln a
    dlnH_dlna = -1.5 * (1.0 + w0 + wa - w0 * Om - wa * Om)  # simplified
    # Actually: = -3/2 * [Om_m + u * Om_de] / [Om_m + Om_de] = -3/2 * [1 + (u-1)*Om_de]
    dlnH_dlna = -1.5 * (1.0 + (u - 1.0) * Om_de)
    return (1.5 * Om - (1.5 + dlnH_dlna))  # This gives f from the algebraic relation? No...
    # Wait, the ODE is first order for f: f' = 1.5*Om - (1.5 + dlnH/dlna)*f
    # This is not algebraic. I need to solve the ODE numerically.
    # Let me return a placeholder and use a different approach.
    pass

def _solve_growth_ODE(a_eval, Om0, w0, wa, n_steps=2000):
    """Solve the growth ODE numerically to get f_ref and D_ref at a_eval."""
    # Integrate from a=1 (where D=1, f=f(1)) down to a_eval
    # f' = d f / d ln a = 1.5 * Om(a) - (1.5 + dlnH/dlna) * f
    # D' = d D / d ln a = f * D
    # Initial conditions at a=1: D(1)=1, f(1) = Om0^0.55 (standard)
    
    a_eval = np.asarray(a_eval, dtype=float)
    a_min = np.min(a_eval)
    a_max = np.max(a_eval)
    
    # We integrate in ln a from ln(1)=0 to ln(a_min) (negative)
    # Use a dense grid
    n = 4000
    ln_a_grid = np.linspace(0, np.log(a_min), n)
    ln_a_grid_eval = np.log(a_eval)
    
    f_init = Om0**0.55
    D_init = 1.0
    
    # Use simple RK4
    def rhs(ln_a, f, D):
        a = np.exp(ln_a)
        Om = _Om_of_a(a, Om0, w0, wa)
        Om_de = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        dlnH_dlna = -1.5 * (1.0 + (u - 1.0) * Om_de)
        df_dlna = 1.5 * Om - (1.5 + dlnH_dlna) * f
        dD_dlna = f * D
        return df_dlna, dD_dlna
    
    f_arr = np.zeros(n)
    D_arr = np.zeros(n)
    f_arr[0] = f_init
    D_arr[0] = D_init
    
    h = (ln_a_grid[1] - ln_a_grid[0])
    for i in range(n - 1):
        ln_a = ln_a_grid[i]
        f = f_arr[i]
        D = D_arr[i]
        
        k1f, k1D = rhs(ln_a, f, D)
        k2f, k2D = rhs(ln_a + h/2, f + h/2*k1f, D + h/2*k1D)
        k3f, k3D = rhs(ln_a + h/2, f + h/2*k2f, D + h/2*k2D)
        k4f, k4D = rhs(ln_a + h, f + h*k3f, D + h*k3D)
        
        f_arr[i+1] = f + h/6 * (k1f + 2*k2f + 2*k3f + k4f)
        D_arr[i+1] = D + h/6 * (k1D + 2*k2D + 2*k3D + k4D)
    
    # Interpolate to get f and D at a_eval
    f_ref = np.interp(ln_a_grid_eval, ln_a_grid, f_arr)
    D_ref = np.interp(ln_a_grid_eval, ln_a_grid, D_arr)
    D_norm = D_ref / D_ref[0]  # D(1) = 1 already
    return f_ref, D_norm

def _compute_errors(c, Om0, w0, wa, a_vals):
    """Compute max relative error of f and D for given coefficients."""
    Om = _Om_of_a(a_vals, Om0, w0, wa)
    g = _gamma_of_a(a_vals, c, Om0, w0, wa)
    
    # Check gamma bounds
    if np.any(g < 0.1) or np.any(g > 1.2):
        return 1e10
    
    f_approx = Om**g
    
    # Reference
    f_ref, D_ref = _solve_growth_ODE(a_vals, Om0, w0, wa)
    
    # D_approx: exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # = exp(-integral from ln(a) to 0 of Om(exp(x))^gamma(exp(x)) dx)
    # Let's compute this numerically
    ln_a_vals = np.log(a_vals)
    # For each a, integrate from ln_a to 0
    D_approx = np.zeros_like(a_vals)
    for i, la in enumerate(ln_a_vals):
        # Integrate from la to 0 of f(exp(x)) dx
        x_grid = np.linspace(la, 0, 200)
        a_grid = np.exp(x_grid)
        Om_grid = _Om_of_a(a_grid, Om0, w0, wa)
        g_grid = _gamma_of_a(a_grid, c, Om0, w0, wa)
        f_grid = Om_grid**g_grid
        integral = np.trapz(f_grid, x_grid)
        D_approx[i] = np.exp(-integral)
    
    # Relative errors
    err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
    err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
    
    max_err = np.max(np.concatenate([err_f, err_D]))
    return max_err

def _objective(c, validation_models, a_grid):
    """Max relative error across all validation models."""
    max_err = 0.0
    for Om0, w0, wa in validation_models:
        err = _compute_errors(c, Om0, w0, wa, a_grid)
        if err > max_err:
            max_err = err
    return max_err

# Pre-compute validation models and a grid
# Typical validation cosmologies for CPL dark energy
_VALIDATION_MODELS = [
    (0.3, -1.0, 0.0),      # LCDM
    (0.3, -1.0, 0.3),      # CPL with wa=0.3
    (0.3, -0.9, 0.2),      # mild DE
    (0.25, -1.0, 0.0),     # lower Om0
    (0.35, -1.0, 0.0),     # higher Om0
    (0.3, -0.8, 0.1),      # different w0
    (0.3, -1.2, 0.0),      # phantom-ish
    (0.3, -1.0, -0.2),     # negative wa
]

_A_GRID = np.linspace(0.05, 1.0, 50)

def construct():
    global _START_TIME
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_BUDGET:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Start with the seed
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    try:
        # Use Nelder-Mead to minimize max error
        res = minimize(
            lambda c: _objective(c, _VALIDATION_MODELS, _A_GRID),
            c0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        c_best = res.x
        # Clamp to [-10, 10]
        c_best = np.clip(c_best, -10.0, 10.0)
        
        elapsed = time.time() - _START_TIME
        if elapsed > _TIME_BUDGET:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Polish with tighter settings if time allows
        if elapsed < _TIME_BUDGET - 20:
            res2 = minimize(
                lambda c: _objective(c, _VALIDATION_MODELS, _A_GRID),
                c_best,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            c_best = np.clip(res2.x, -10.0, 10.0)
        
        return list(c_best)
        
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
