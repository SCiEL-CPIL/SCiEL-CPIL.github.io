import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

_start_time = time.time()
_DEADLINE = 75.0

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def _H2_of_a(a, Om_m, w0, wa):
    """Compute H²(a)/H0² for CPL dark energy. Vectorized."""
    Om_de = 1.0 - Om_m
    w = w0 + wa * (1.0 - a)
    rho_de = Om_de * np.power(a, -3.0 * (1.0 + w))
    rho_m = Om_m / a**3
    return rho_m + rho_de

def _Om_m_of_a(a, Om_m, w0, wa):
    """Compute Om_m(a) = rho_m / rho_total. Vectorized."""
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

def _compute_D_approx_fast(a_grid, coeffs, Om_m, w0, wa, x_fine, n_quad):
    """Fast vectorized computation of D(a)/D(1) using the gamma formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = a_grid
    D_approx = np.ones_like(a)
    
    # Precompute on the fine grid
    a_fine = np.exp(x_fine)
    Om_ma_fine = _Om_m_of_a(a_fine, Om_m, w0, wa)
    q_fine = 1.0 - Om_ma_fine
    u_fine = 1.0 + w0 + wa * (1.0 - a_fine)
    gamma_fine = c0 + c1*q_fine + c2*q_fine**2 + c3*u_fine + c4*wa*a_fine + c5*u_fine*q_fine
    
    # Check gamma bounds
    if np.any(gamma_fine < 0.1) or np.any(gamma_fine > 1.2):
        return None
    
    f_fine = np.power(Om_ma_fine, gamma_fine)
    
    # Cumulative trapezoidal integral from left to right
    # integral from x_min to x = cumtrapz
    dx = np.diff(x_fine)
    trapz = 0.5 * (f_fine[:-1] + f_fine[1:]) * dx
    cum_int = np.concatenate([[0], np.cumsum(trapz)])
    total_int = cum_int[-1]
    
    # For each a_i, compute integral from ln(a_i) to 0 = total_int - int_at_ln(a_i)
    x_i = np.log(a_grid)
    
    # Vectorized interpolation of cumulative integral
    # np.interp expects increasing xp, and x_fine is increasing
    int_at_xi = np.interp(x_i, x_fine, cum_int)
    
    integral = total_int - int_at_xi
    D_approx = np.exp(-integral)
    
    # Set D(1) = 1 exactly
    D_approx[np.abs(a - 1.0) < 1e-10] = 1.0
    
    return D_approx

def construct():
    global _start_time
    _start_time = time.time()
    
    default_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # Representative cosmologies - focus on challenging ones
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
            (0.25, -1.0, 0.2),
            (0.35, -1.0, -0.2),
            (0.30, -1.0, 0.1),
            (0.28, -0.95, 0.15),
            (0.32, -1.05, -0.15),
        ]
        
        # Scale factor grid - dense enough for accurate integration
        a_grid = np.logspace(np.log10(0.05), 0.0, 50)
        
        # Fine grid for D integration - shared across all cosmologies
        n_quad = 200
        x_fine_global = np.linspace(np.log(a_grid.min()), 0.0, n_quad)
        
        # Precompute reference data
        ref_data = []
        for Om_m, w0, wa in cosmologies:
            if _check_deadline():
                break
            f_ref, D_ref = _solve_growth(a_grid, Om_m, w0, wa)
            if f_ref is not None:
                ref_data.append((a_grid, f_ref, D_ref, Om_m, w0, wa))
        
        if not ref_data:
            return default_coeffs
        
        def compute_max_rel_err(coeffs):
            """Compute the maximum relative error across all cosmologies and a values."""
            c0, c1, c2, c3, c4, c5 = coeffs
            max_err = 0.0
            
            for a_g, f_ref, D_ref, Om_m, w0, wa in ref_data:
                a = a_g
                Om_ma = _Om_m_of_a(a, Om_m, w0, wa)
                q = 1.0 - Om_ma
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds
                if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                    return 100.0
                
                f_approx = np.power(Om_ma, gamma)
                rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
                max_err = max(max_err, np.max(rel_err_f))
                
                # Compute D approximation
                D_approx = _compute_D_approx_fast(a, coeffs, Om_m, w0, wa, x_fine_global, n_quad)
                if D_approx is None:
                    return 100.0
                
                rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
                max_err = max(max_err, np.max(rel_err_D))
            
            return max_err
        
        # Two-stage optimization
        # Stage 1: Coarse grid search
        best_coeffs = default_coeffs.copy()
        best_err = compute_max_rel_err(best_coeffs)
        
        if not _check_deadline():
            # Coarse grid
            grid_c0 = np.linspace(0.4, 0.7, 7)
            grid_c1 = np.linspace(-0.5, 0.5, 5)
            grid_c2 = np.linspace(-0.3, 0.3, 5)
            grid_c3 = np.linspace(-0.3, 0.3, 5)
            grid_c4 = np.linspace(-0.3, 0.3, 5)
            grid_c5 = np.linspace(-0.3, 0.3, 5)
            
            # Limit the search to keep within time budget
            count = 0
            for c0 in grid_c0:
                for c1 in grid_c1:
                    for c2 in grid_c2:
                        for c3 in grid_c3:
                            for c4 in grid_c4:
                                for c5 in grid_c5:
                                    count += 1
                                    if count % 200 == 0 and _check_deadline():
                                        break
                                    coeffs = [c0, c1, c2, c3, c4, c5]
                                    err = compute_max_rel_err(coeffs)
                                    if err < best_err:
                                        best_err = err
                                        best_coeffs = coeffs
                                if _check_deadline():
                                    break
                            if _check_deadline():
                                break
                        if _check_deadline():
                            break
                    if _check_deadline():
                        break
                if _check_deadline():
                    break
            if _check_deadline():
                return [float(c) for c in best_coeffs]
        
        # Stage 2: Fine-tuning with Nelder-Mead
        if not _check_deadline():
            def objective(coeffs):
                if _check_deadline():
                    return 1000.0
                return compute_max_rel_err(coeffs)
            
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                bounds=[(-10, 10)] * 6,
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < best_err:
                best_err = result.fun
                best_coeffs = result.x
        
        # Stage 3: Additional local refinement with multiple restarts
        if not _check_deadline():
            for restart in range(3):
                # Perturb the best solution
                perturb = np.random.normal(0, 0.05, 6)
                x0 = np.clip(best_coeffs + perturb, -10, 10)
                
                def objective2(coeffs):
                    if _check_deadline():
                        return 1000.0
                    return compute_max_rel_err(coeffs)
                
                result = minimize(
                    objective2,
                    x0,
                    method='Nelder-Mead',
                    bounds=[(-10, 10)] * 6,
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                
                if result.fun < best_err:
                    best_err = result.fun
                    best_coeffs = result.x
                
                if _check_deadline():
                    break
        
        # Verify gamma bounds on a dense grid for all cosmologies
        for Om_m, w0, wa in cosmologies:
            a_test = np.linspace(0.05, 1.0, 500)
            Om_ma = _Om_m_of_a(a_test, Om_m, w0, wa)
            q = 1.0 - Om_ma
            u = 1.0 + w0 + wa * (1.0 - a_test)
            gamma = (best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + 
                     best_coeffs[3]*u + best_coeffs[4]*wa*a_test + best_coeffs[5]*u*q)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return default_coeffs
        
        if not all(-10 <= c <= 10 for c in best_coeffs):
            return default_coeffs
        
        return [float(c) for c in best_coeffs]
    
    except Exception:
        return default_coeffs
