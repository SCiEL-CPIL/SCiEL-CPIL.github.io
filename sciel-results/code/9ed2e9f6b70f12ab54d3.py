"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

def construct():
    """Return six coefficients for the growth-index ansatz."""
    # Deadline guard: 90s wall-clock limit, use 80s for safety
    t0 = time.time()
    deadline = t0 + 80.0
    
    # Precompute reference data
    ref_data = _make_reference_data()
    
    if time.time() > deadline - 5:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: constant gamma
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _objective(best_c, ref_data)
    
    # Try multiple starting points with Nelder-Mead
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, -0.05, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.02, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
    ]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _objective, start, args=(ref_data,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.success and res.fun < best_score:
                best_c = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective, best_c, args=(ref_data,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.success and res.fun < best_score:
                best_c = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    return list(best_c)

def _compute_Om_m(Om0, w0, wa, a):
    """Compute matter density parameter Om_m(a) = Om0 * a^-3 / E^2(a)."""
    Om_m = Om0 * a**(-3)
    Om_de0 = 1.0 - Om0
    w_eff = w0 + wa * (1.0 - a)
    Om_de = Om_de0 * a**(-3.0 * (1.0 + w_eff))
    E2 = Om_m + Om_de
    return Om_m / E2

def _make_reference_data():
    """Generate reference f(a) and D(a) for validation cosmologies."""
    cosmo_params = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.3),   # CPL
        (0.30, -0.9, 0.0),
        (0.30, -0.9, 0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -1.2, 0.5),   # Phantom
        (0.30, -0.5, 0.0),   # w0 > -1
        (0.40, -1.0, 0.3),
        (0.20, -1.0, 0.3),
        (0.30, -1.1, 0.2),
        (0.28, -0.8, 0.4),
    ]
    
    a_grid = np.linspace(0.1, 1.0, 200)
    
    ref_data = []
    for Om0, w0, wa in cosmo_params:
        try:
            f_ref, D_ref = _generate_reference(Om0, w0, wa, a_grid)
            ref_data.append((Om0, w0, wa, f_ref, D_ref))
        except Exception:
            pass
    
    return ref_data

def _generate_reference(Om0, w0, wa, a_grid):
    """Generate reference f(a) and D(a) via ODE integration."""
    def ode_f(a, y):
        f = y[0]
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        return [(1.5 * Om_m - f * (f + 1)) / a]
    
    # Integrate from a=0.01 to a=1.0 to find f(1)
    try:
        sol = solve_ivp(ode_f, [0.01, 1.0], [1.0], method='RK45', rtol=1e-9, atol=1e-11)
        if not sol.success:
            f1 = Om0**0.55
        else:
            f1 = sol.y[0, -1]
    except Exception:
        f1 = Om0**0.55
    
    # Integrate over full range with dense output
    a_min = a_grid[0]
    a_max = a_grid[-1]
    
    a_eval = np.sort(np.unique(np.concatenate([a_grid, [1.0]])))
    try:
        sol = solve_ivp(ode_f, [a_min, a_max], [f1], method='RK45', 
                       t_eval=a_eval, rtol=1e-9, atol=1e-11)
        if not sol.success:
            # Fallback: interpolate from separate integrations
            sol_down = solve_ivp(ode_f, [1.0, a_min], [f1], method='RK45', 
                                t_eval=a_eval[a_eval <= 1.0], rtol=1e-9, atol=1e-11)
            sol_up = solve_ivp(ode_f, [1.0, a_max], [f1], method='RK45', 
                              t_eval=a_eval[a_eval >= 1.0], rtol=1e-9, atol=1e-11)
            a_all = np.concatenate([sol_down.t, sol_up.t[1:]])
            f_all = np.concatenate([sol_down.y[0], sol_up.y[0][1:]])
            sort_idx = np.argsort(a_all)
            a_sorted = a_all[sort_idx]
            f_sorted = f_all[sort_idx]
            f_ref = np.interp(a_grid, a_sorted, f_sorted)
        else:
            f_ref = np.interp(a_grid, sol.t, sol.y[0])
    except Exception:
        f_ref = np.power(_compute_Om_m(Om0, w0, wa, a_grid), 0.55)
    
    # Compute D(a) = exp(integral_1^a f(a')/a' da')
    a_dense = np.linspace(a_min, a_max, 1000)
    f_dense = np.interp(a_dense, a_grid, f_ref)
    integrand = f_dense / a_dense
    
    grid = np.unique(np.concatenate([a_dense, [1.0]]))
    f_grid = np.interp(grid, a_grid, f_ref)
    int_grid = f_grid / grid
    
    C = cumulative_trapezoid(int_grid, grid, initial=0)
    idx_1 = np.argmin(np.abs(grid - 1.0))
    C1 = C[idx_1]
    
    I_a = np.interp(a_grid, grid, C - C1)
    D_ref = np.exp(I_a)
    
    return f_ref, D_ref

def _objective(c, ref_data):
    """Compute 100 * max relative error of f and D across all cosmologies."""
    c0, c1, c2, c3, c4, c5 = c
    max_err = 0.0
    
    for Om0, w0, wa, f_ref, D_ref in ref_data:
        a = np.linspace(0.1, 1.0, 200)
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        # Check gamma constraints
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_model = np.power(Om_m, gamma)
        
        # Compute D_model
        integrand = f_model / a
        C = cumulative_trapezoid(integrand, a, initial=0)
        idx_1 = np.argmin(np.abs(a - 1.0))
        C1 = C[idx_1]
        I_a = C - C1
        D_model = np.exp(I_a)
        
        # Relative errors
        err_f = np.abs(f_model - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        err_D = np.abs(D_model - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return 100.0 * max_err
# EVOLVE-BLOCK-END
