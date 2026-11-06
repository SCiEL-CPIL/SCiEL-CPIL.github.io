import numpy as np
import time
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

_start_time = time.time()
_DEADLINE = 72.0  # 72s out of 90s, leaving 18s margin

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

def _compute_D_approx(a_grid, coeffs, precomp):
    """Compute D(a)/D(1) using precomputed integration grid."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = a_grid
    D_approx = np.ones_like(a)
    
    x_fine, a_fine, Om_ma_fine, q_fine, u_fine, wa_fine = precomp
    gamma_fine = c0 + c1*q_fine + c2*q_fine**2 + c3*u_fine + c4*wa_fine + c5*u_fine*q_fine
    
    if np.any(gamma_fine < 0.1) or np.any(gamma_fine > 1.2):
        return None
    
    f_fine = np.power(Om_ma_fine, gamma_fine)
    
    dx = x_fine[1] - x_fine[0]
    cum_int = np.cumsum(f_fine) * dx
    cum_int = np.concatenate([[0.0], cum_int])
    total_int = cum_int[-1]
    
    # Vectorized interpolation
    x_i = np.log(a)
    idx = np.searchsorted(x_fine, x_i)
    idx = np.clip(idx, 1, len(x_fine) - 1)
    
    x0_fine = x_fine[idx - 1]
    x1_fine = x_fine[idx]
    c0_int = cum_int[idx - 1]
    c1_int = cum_int[idx]
    
    denom = x1_fine - x0_fine
    denom = np.where(np.abs(denom) < 1e-15, 1.0, denom)
    frac = (x_i - x0_fine) / denom
    frac = np.clip(frac, 0.0, 1.0)
    
    int_at_x = c0_int + frac * (c1_int - c0_int)
    integral = total_int - int_at_x
    
    # For a close to 1, D should be ~1
    mask = a >= 0.9999
    if np.any(mask):
        integral[mask] = 0.0
    
    D_approx = np.exp(-integral)
    return D_approx

def _precompute_integration_grid(Om_m, w0, wa, a_min, n_quad=200):
    """Precompute the fine grid data for D integration."""
    x_fine = np.linspace(np.log(a_min), 0.0, n_quad)
    a_fine = np.exp(x_fine)
    Om_ma_fine = _Om_m_of_a(a_fine, Om_m, w0, wa)
    q_fine = 1.0 - Om_ma_fine
    u_fine = 1.0 + w0 + wa * (1.0 - a_fine)
    wa_fine = wa * a_fine  # c4 * wa * a
    return (x_fine, a_fine, Om_ma_fine, q_fine, u_fine, wa_fine)

def _max_rel_error(coeffs, ref_data, precomp_list):
    """Compute the maximum relative error across all observables and cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for i, (a_g, f_ref, D_ref, Om_m, w0, wa) in enumerate(ref_data):
        a = a_g
        Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
        q = 1.0 - Om_ma
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 100.0
        
        f_approx = np.power(Om_ma, gamma)
        rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_err = max(max_err, np.max(rel_err_f))
        
        D_approx = _compute_D_approx(a, coeffs, precomp_list[i])
        if D_approx is None:
            return 100.0
        
        rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(rel_err_D))
    
    return max_err

def construct():
    global _start_time
    _start_time = time.time()
    
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Broader cosmology coverage
        cosmologies = [
            (0.25, -1.0, 0.0),
            (0.27, -1.0, 0.0),
            (0.30, -1.0, 0.0),
            (0.33, -1.0, 0.0),
            (0.35, -1.0, 0.0),
            (0.25, -1.0, 0.3),
            (0.30, -1.0, 0.3),
            (0.35, -1.0, 0.3),
            (0.25, -1.0, -0.3),
            (0.30, -1.0, -0.3),
            (0.35, -1.0, -0.3),
            (0.27, -1.1, 0.0),
            (0.30, -0.9, 0.0),
            (0.30, -1.2, 0.5),
            (0.30, -0.8, -0.5),
            (0.20, -1.0, 0.0),
            (0.40, -1.0, 0.0),
            (0.27, -1.0, 0.5),
            (0.30, -1.0, -0.5),
        ]
        
        # Denser scale factor grid
        a_grid = np.logspace(np.log10(0.03), 0.0, 40)
        
        # Precompute reference data
        ref_data = []
        precomp_list = []
        for Om_m, w0, wa in cosmologies:
            f_ref, D_ref = _solve_growth(a_grid, Om_m, w0, wa)
            if f_ref is not None:
                ref_data.append((a_grid, f_ref, D_ref, Om_m, w0, wa))
                precomp = _precompute_integration_grid(Om_m, w0, wa, a_grid.min(), n_quad=200)
                precomp_list.append(precomp)
        
        if not ref_data:
            return default_coeffs
        
        bounds = ([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
        
        # Multi-start with diverse initial points
        initial_points = [
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.55, 0.0, -0.1, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, -0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, -0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.55, 0.0, 0.0, 0.0, 0.0, -0.1],
            [0.50, 0.15, 0.05, 0.05, 0.0, 0.0],
            [0.60, -0.1, -0.05, 0.1, 0.05, 0.05],
        ]
        
        best_coeffs = default_coeffs
        best_err = 100.0
        
        # Objective for least_squares: use smooth max via log-sum-exp
        def objective(coeffs):
            if _check_deadline():
                n_total = sum(len(a) for a, _, _, _, _ in ref_data)
                return [100.0] * (2 * n_total)
            
            c0, c1, c2, c3, c4, c5 = coeffs
            residuals = []
            
            for i, (a_g, f_ref, D_ref, Om_m, w0, wa) in enumerate(ref_data):
                a = a_g
                Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
                q = 1.0 - Om_ma
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    residuals.extend([100.0] * len(a) * 2)
                    continue
                
                f_approx = np.power(Om_ma, gamma)
                rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
                residuals.extend(rel_err_f)
                
                D_approx = _compute_D_approx(a, coeffs, precomp_list[i])
                if D_approx is None:
                    residuals.extend([100.0] * len(a))
                    continue
                
                rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
                residuals.extend(rel_err_D)
            
            return residuals
        
        # Try each initial point
        for x0 in initial_points:
            if _check_deadline():
                break
            
            try:
                result = least_squares(
                    objective, x0, bounds=bounds,
                    method='trf', max_nfev=200, xtol=1e-12, ftol=1e-12
                )
                
                # Evaluate the actual max error for this result
                err = _max_rel_error(result.x, ref_data, precomp_list)
                if err < best_err:
                    best_err = err
                    best_coeffs = result.x
            except Exception:
                continue
        
        # Verify gamma bounds on a dense grid for all cosmologies
        for Om_m, w0, wa in cosmologies:
            a_test = np.linspace(0.03, 1.0, 300)
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
