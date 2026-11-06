import numpy as np
import time
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

_start_time = time.time()
_DEADLINE = 75.0  # 75s out of 90s, leaving 15s margin

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def _H2_of_a(a, Om_m, w0, wa):
    """Compute H²(a)/H0² for CPL dark energy."""
    Om_de = 1.0 - Om_m
    w = w0 + wa * (1.0 - a)
    rho_de = Om_de * np.power(a, -3.0 * (1.0 + w))
    rho_m = Om_m / a**3
    return rho_m + rho_de

def _Om_m_of_a(a, Om_m, w0, wa):
    """Compute Om_m(a) = rho_m / rho_total."""
    H2 = _H2_of_a(a, Om_m, w0, wa)
    rho_m = Om_m / a**3
    return rho_m / H2

def _growth_ode(x, y, Om_m, w0, wa):
    """ODE for growth factor in terms of x = ln(a)."""
    a = np.exp(x)
    D, Dp = y
    H2 = _H2_of_a(a, Om_m, w0, wa)
    Om_de = 1.0 - Om_m
    w = w0 + wa * (1.0 - a)
    rho_m = Om_m / a**3
    rho_de = Om_de * np.power(a, -3.0 * (1.0 + w))
    drho_m_dx = -3.0 * rho_m
    drho_de_dx = rho_de * (-3.0 * (1.0 + w) + 3.0 * wa * a)
    dH2_dx = drho_m_dx + drho_de_dx
    dlnH_dx = 0.5 * dH2_dx / H2
    Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
    Dpp = -(3.0 + dlnH_dx) * Dp + 1.5 * Om_ma * D
    return [Dp, Dpp]

def _solve_growth(a_grid, Om_m, w0, wa):
    """Solve the growth ODE on a_grid for given cosmology."""
    x_grid = np.log(a_grid)
    a_min = a_grid.min()
    x_min = np.log(a_min)
    x_max = 0.0
    
    D0 = np.exp(1.5 * x_min)
    Dp0 = 1.5 * D0
    
    sol = solve_ivp(
        _growth_ode, [x_min, x_max], [D0, Dp0],
        args=(Om_m, w0, wa),
        t_eval=x_grid,
        rtol=1e-10, atol=1e-12,
        method='DOP853'
    )
    
    if not sol.success:
        return None, None
    
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    f_vals = Dp_vals / D_vals
    D_at_1 = D_vals[-1]
    D_norm = D_vals / D_at_1
    return f_vals, D_norm

def _compute_D_approx_vectorized(a_grid, coeffs, Om_m, w0, wa, n_quad=50):
    """Vectorized computation of D(a)/D(1) using the gamma formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = a_grid
    D_approx = np.ones_like(a)
    
    # For each a_i, compute integral from ln(a_i) to 0 of f(exp(x)) dx
    # We can do this by precomputing f on a dense grid and using cumulative integration
    # But since the integration range differs for each a_i, let's use a different approach
    
    # Precompute on a fine grid in x-space
    x_fine = np.linspace(np.log(a.min()), 0.0, n_quad)
    a_fine = np.exp(x_fine)
    Om_ma_fine = _Om_m_of_a(a_fine, Om_m, w0, wa)
    q_fine = 1.0 - Om_ma_fine
    u_fine = 1.0 + w0 + wa * (1.0 - a_fine)
    gamma_fine = c0 + c1*q_fine + c2*q_fine**2 + c3*u_fine + c4*wa*a_fine + c5*u_fine*q_fine
    
    # Check gamma bounds
    if np.any(gamma_fine < 0.1) or np.any(gamma_fine > 1.2):
        return None
    
    f_fine = np.power(Om_ma_fine, gamma_fine)
    
    # Compute cumulative integral from the end (x=0) backwards
    # integral from x to 0 = total_integral - integral from x_min to x
    dx = x_fine[1] - x_fine[0]
    # Cumulative integral from left
    cum_int = np.cumsum(f_fine) * dx
    cum_int = np.concatenate([[0], cum_int])
    total_int = cum_int[-1]
    
    # For each a_i, find where ln(a_i) falls in x_fine and interpolate
    for i, ai in enumerate(a):
        if ai >= 0.9999:
            D_approx[i] = 1.0
        else:
            x_i = np.log(ai)
            # Find index in x_fine
            idx = np.searchsorted(x_fine, x_i)
            if idx >= len(x_fine):
                integral = total_int
            elif idx == 0:
                integral = total_int
            else:
                # Interpolate the cumulative integral
                x0_fine, x1_fine = x_fine[idx-1], x_fine[idx]
                c0_int, c1_int = cum_int[idx-1], cum_int[idx]
                if x1_fine > x0_fine:
                    frac = (x_i - x0_fine) / (x1_fine - x0_fine)
                    int_at_x = c0_int + frac * (c1_int - c0_int)
                else:
                    int_at_x = c0_int
                integral = total_int - int_at_x
            D_approx[i] = np.exp(-integral)
    
    return D_approx

def construct():
    global _start_time
    _start_time = time.time()
    
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Representative cosmologies covering the parameter space
        cosmologies = [
            (0.27, -1.0, 0.0),
            (0.30, -1.0, 0.0),
            (0.33, -1.0, 0.0),
            (0.27, -1.0, 0.3),
            (0.30, -1.0, 0.3),
            (0.33, -1.0, 0.3),
            (0.27, -1.0, -0.3),
            (0.30, -1.0, -0.3),
            (0.33, -1.0, -0.3),
            (0.25, -1.1, 0.0),
            (0.35, -0.9, 0.0),
            (0.30, -1.2, 0.5),
            (0.30, -0.8, -0.5),
            (0.20, -1.0, 0.0),
            (0.40, -1.0, 0.0),
        ]
        
        # Scale factor grid
        a_grid = np.logspace(np.log10(0.05), 0.0, 25)
        
        # Precompute reference data
        ref_data = []
        for Om_m, w0, wa in cosmologies:
            f_ref, D_ref = _solve_growth(a_grid, Om_m, w0, wa)
            if f_ref is not None:
                ref_data.append((a_grid, f_ref, D_ref, Om_m, w0, wa))
        
        if not ref_data:
            return default_coeffs
        
        # Precompute the fine grid for D integration
        n_quad = 100
        x_fine_global = np.linspace(np.log(a_grid.min()), 0.0, n_quad)
        a_fine_global = np.exp(x_fine_global)
        
        def objective(coeffs):
            if _check_deadline():
                # Return large residuals to stop optimization
                n_total = sum(len(a) for a, _, _, _, _ in ref_data)
                return [100.0] * (2 * n_total)
            
            c0, c1, c2, c3, c4, c5 = coeffs
            residuals = []
            
            for a_g, f_ref, D_ref, Om_m, w0, wa in ref_data:
                a = a_g
                Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
                q = 1.0 - Om_ma
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds on the evaluation grid
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    residuals.extend([100.0] * len(a) * 2)
                    continue
                
                f_approx = np.power(Om_ma, gamma)
                rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
                residuals.extend(rel_err_f)
                
                # Compute D approximation
                D_approx = _compute_D_approx_vectorized(a, coeffs, Om_m, w0, wa, n_quad=50)
                if D_approx is None:
                    residuals.extend([100.0] * len(a))
                    continue
                
                rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
                residuals.extend(rel_err_D)
            
            return residuals
        
        bounds = ([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
        x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        result = least_squares(
            objective, x0, bounds=bounds,
            method='trf', max_nfev=300, xtol=1e-10, ftol=1e-10
        )
        
        best_coeffs = result.x
        
        # Verify gamma bounds on a dense grid for all cosmologies
        for Om_m, w0, wa in cosmologies:
            a_test = np.linspace(0.05, 1.0, 200)
            Om_ma = _Om_m_of_a(a_test, Om_m, w0, wa)
            q = 1.0 - Om_ma
            u = 1.0 + w0 + wa * (1.0 - a_test)
            gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + best_coeffs[3]*u + best_coeffs[4]*wa*a_test + best_coeffs[5]*u*q
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return default_coeffs
        
        if not all(-10 <= c <= 10 for c in best_coeffs):
            return default_coeffs
        
        return [float(c) for c in best_coeffs]
    
    except Exception:
        return default_coeffs
