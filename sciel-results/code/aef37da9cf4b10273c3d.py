"""Improved growth-index ansatz with basis-transformed optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 80.0

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    
    We optimize these coefficients to minimize worst-case relative error
    across validation cosmologies for both f and D.
    """
    # Seed: constant gamma = 0.55
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = np.inf
    
    # Define validation cosmologies (typical CPL dark energy models)
    # (Om0, w0, wa)
    val_cosmos = [
        (0.3, -1.0, 0.0),      # Lambda CDM
        (0.3, -1.0, 0.1),      # mild phantom/accel
        (0.3, -1.0, -0.1),     # quintessence-like
        (0.3, -0.9, 0.0),      # early-decaying
        (0.3, -1.1, 0.0),      # phantom
        (0.3, -1.0, 0.2),      # strong wa
        (0.28, -1.0, 0.0),     # slightly lower Om0
        (0.32, -1.0, 0.0),     # slightly higher Om0
        (0.3, -0.8, 0.3),      # more extreme
        (0.3, -1.2, -0.2),     # phantom with negative wa
    ]
    
    # Scale factor grid for evaluation
    a_grid = np.logspace(np.log10(0.1), 0, 200)  # 0.1 to 1
    
    def _Om_a_num(a_val, Om0, w0, wa):
        """Numerical computation of Om(a) at a single point."""
        if a_val <= 0:
            return Om0
        # Integrate d ln rho_de / d ln a = -3(1+w(a))
        # rho_de(a) = rho_de(1) * exp(-3 * integral_1^a (1+w(a'))/a' da')
        # Change variable: x = ln a', a' = exp(x), da'/a' = dx
        # integral_1^a (1+w(a'))/a' da' = integral_0^{ln a} (1+w(exp(x))) dx
        # w(exp(x)) = w0 + wa*(1-exp(x))
        # 1+w(exp(x)) = 1+w0+wa*(1-exp(x))
        
        if abs(a_val - 1.0) < 1e-15:
            integral = 0.0
        else:
            integral = np.log(a_val)*(1+w0+wa) - wa*(a_val-1)
        
        rho_de_ratio = np.exp(-3 * integral)
        rho_m_ratio = a_val**(-3)
        Om_a = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1-Om0) * rho_de_ratio)
        return Om_a
    
    def _Om_a_vec(a_arr, Om0, w0, wa):
        """Vectorized Om(a) computation."""
        return np.array([_Om_a_num(ai, Om0, w0, wa) for ai in a_arr])
    
    def _compute_gammas(a, coeff, Om0, w0, wa):
        """Compute gamma values on a grid."""
        Om_arr = _Om_a_vec(a, Om0, w0, wa)
        q = 1.0 - Om_arr
        u = 1.0 + w0 + wa * (1.0 - a)
        c0, c1, c2, c3, c4, c5 = coeff
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        return gamma, Om_arr, q, u
    
    def _compute_errors(coeff, a_grid, val_cosmos):
        """Compute max relative error for f and D over all validation cosmologies."""
        c0, c1, c2, c3, c4, c5 = coeff
        max_err_f = 0.0
        max_err_D = 0.0
        
        for (Om0, w0, wa) in val_cosmos:
            # Compute Om(a) on grid
            Om_arr = _Om_a_vec(a_grid, Om0, w0, wa)
            
            # Compute q, u, gamma
            q = 1.0 - Om_arr
            u = 1.0 + w0 + wa * (1.0 - a_grid)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return np.inf  # Invalid
            
            # f = Om(a)^gamma
            f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
            
            # Reference f: use the exact growth index from the ODE
            # f' = f * (f + 3/2) - 3/2 * Om(a)
            # where ' = d/d ln a
            
            # Integrate the ODE for f_ref from a=1 backward to a=0.1
            n = len(a_grid)
            f_ref = np.zeros(n)
            
            # Initial condition at a=1: f(1) ≈ Om0^0.55
            f_ref[-1] = Om0**0.55
            
            # Integrate backward using RK4
            for i in range(n-2, -1, -1):
                x1 = np.log(a_grid[i+1])
                x2 = np.log(a_grid[i])
                dx = x2 - x1  # negative
                
                def f_ode(x, f):
                    a_x = np.exp(x)
                    Om_x = _Om_a_num(a_x, Om0, w0, wa)
                    return f*(f + 1.5) - 1.5*Om_x
                
                # RK4 step
                k1 = f_ode(x1, f_ref[i+1])
                k2 = f_ode(x1 + dx/2, f_ref[i+1] + dx/2 * k1)
                k3 = f_ode(x1 + dx/2, f_ref[i+1] + dx/2 * k2)
                k4 = f_ode(x1 + dx, f_ref[i+1] + dx * k3)
                
                f_ref[i] = f_ref[i+1] + dx/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Compute D from f: D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)
            D_ours = np.ones(n)
            D_ref_norm = np.ones(n)
            
            # Use cumulative trapezoidal integration
            x_vals = np.log(a_grid)
            
            for i in range(n):
                if i == n-1:
                    D_ours[i] = 1.0
                    D_ref_norm[i] = 1.0
                    continue
                
                # ∫_{x_i}^0 f dx
                idx_range = slice(i, n)
                x_sub = x_vals[idx_range]
                f_sub_ours = f_ours[idx_range]
                f_sub_ref = f_ref[idx_range]
                
                integral_ours = np.trapz(f_sub_ours, x_sub)
                integral_ref = np.trapz(f_sub_ref, x_sub)
                
                D_ours[i] = np.exp(-integral_ours)
                D_ref_norm[i] = np.exp(-integral_ref)
            
            # Relative errors
            valid = a_grid > 0.15
            if np.any(valid):
                rel_f = np.abs(f_ours[valid] - f_ref[valid]) / np.maximum(np.abs(f_ref[valid]), 1e-10)
                rel_D = np.abs(D_ours[valid] - D_ref_norm[valid]) / np.maximum(np.abs(D_ref_norm[valid]), 1e-10)
                max_err_f = max(max_err_f, np.max(rel_f) if len(rel_f) > 0 else 0)
                max_err_D = max(max_err_D, np.max(rel_D) if len(rel_D) > 0 else 0)
        
        return max_err_f + max_err_D
    
    # Objective function for optimizer
    def objective(coeff):
        if _check_deadline():
            return np.inf
        return _compute_errors(coeff, a_grid, val_cosmos)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            best_coeff,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
        )
        if not _check_deadline() and result.fun < best_score and np.all(np.abs(result.x) <= 10):
            # Check gamma bounds
            test_a = np.logspace(np.log10(0.1), 0, 50)
            valid = True
            for (Om0, w0, wa) in val_cosmos:
                gamma_t, _, _, _ = _compute_gammas(test_a, result.x, Om0, w0, wa)
                if np.any(gamma_t < 0.1) or np.any(gamma_t > 1.2):
                    valid = False
                    break
            if valid:
                best_coeff = result.x
                best_score = result.fun
    except Exception:
        pass
    
    # Try a few more starting points
    if not _check_deadline():
        starting_points = [
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.50, 0.1, 0.05, 0.05, 0.05, 0.05],
        ]
        for sp in starting_points:
            if _check_deadline():
                break
            try:
                res = minimize(
                    objective,
                    np.array(sp),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                )
                if not _check_deadline() and res.fun < best_score and np.all(np.abs(res.x) <= 10):
                    # Quick gamma check
                    test_a = np.logspace(np.log10(0.15), 0, 30)
                    valid = True
                    for (Om0, w0, wa) in val_cosmos[:5]:
                        gamma_t, _, _, _ = _compute_gammas(test_a, res.x, Om0, w0, wa)
                        if np.any(gamma_t < 0.1) or np.any(gamma_t > 1.2):
                            valid = False
                            break
                    if valid:
                        best_coeff = res.x
                        best_score = res.fun
            except Exception:
                pass
    
    # Clamp to [-10, 10]
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END
