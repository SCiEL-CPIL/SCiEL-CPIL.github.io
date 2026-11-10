"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

def _compute_om_a(a, Om0, w0, wa):
    """Compute the matter density parameter Om(a) for a CPL dark energy model."""
    if a <= 0:
        return 0.0
    de_factor = (1 - a)**(-3 * (1 + w0 + wa)) if a < 1 else 1.0
    E2 = Om0 / (a**3) + (1 - Om0) * de_factor
    return (Om0 / (a**3)) / E2

def _solve_growth_ode(a_grid, Om0, w0, wa):
    """Solve the growth ODE and return f = dlnD/dln(a) at each a in a_grid."""
    def ode(x, y):
        a = np.exp(x)
        Om_a = _compute_om_a(a, Om0, w0, wa)
        f, D = y
        # D'' + (3/2)Om(a) D' - (3/2)Om(a) D = 0 where prime is d/dln(a)
        # Let y = [f, D] where f = D'/D
        # f' = D''/D - (D'/D)^2 = -f^2 - (3/2)Om(a)*f + (3/2)Om(a)
        f_prime = -f**2 - 1.5 * Om_a * f + 1.5 * Om_a
        D_prime = f * D
        return [f_prime, D_prime]
    
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])
    x_eval = np.log(a_grid)
    
    # Initial conditions at a=1 (x=0): D(1)=1, f(1) ~ 0.55*Om(1)^0.55 but Om(1)=Om0
    # Actually at a=1, D=1, and f(1) = dlnD/dlna at a=1
    # For LambdaCDM, f(1) ≈ 0.55*0.3^0.55 ≈ 0.55*0.522 ≈ 0.287
    # Let's use f(1) = 0.55 * Om0**0.55 as initial guess
    f_init = 0.55 * Om0**0.55
    D_init = 1.0
    
    sol = solve_ivp(ode, [x_max, x_min], [f_init, D_init], t_eval=x_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        # Fallback: return approximate f
        f_approx = np.zeros_like(a_grid)
        for i, a in enumerate(a_grid):
            Om_a = _compute_om_a(a, Om0, w0, wa)
            f_approx[i] = Om_a**0.55
        return f_approx
    
    f_vals = sol.y[0]
    return f_vals

def _compute_relative_errors(coeffs, test_models, a_values, beta=20.0):
    """Compute a smooth surrogate for the worst-case relative error using log-sum-exp."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    errors = []
    
    for Om0, w0, wa in test_models:
        # Solve ODE once per model
        f_ref = _solve_growth_ode(a_values, Om0, w0, wa)
        
        for i, a in enumerate(a_values):
            Om_a = _compute_om_a(a, Om0, w0, wa)
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            # Check gamma bounds - add large penalty if violated
            if gamma < 0.1 or gamma > 1.2:
                return 10000.0
            
            f_approx = Om_a**gamma
            f_ref_val = f_ref[i]
            
            rel_err_f = abs(f_approx - f_ref_val) / max(abs(f_ref_val), 1e-10)
            errors.append(rel_err_f)
    
    errors = np.array(errors)
    
    # Use log-sum-exp for smooth max approximation
    max_err = np.max(errors)
    if max_err < 1e-10:
        return 0.0
    
    # Soft max: (1/beta) * log(sum(exp(beta * e_i)))
    scaled = beta * errors
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / beta
    
    return soft_max * 100

def construct():
    # Start with the seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Test cosmologies: (Om0, w0, wa)
    test_models = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -0.9, -0.3),   # Quintessence
        (0.3, -1.2, 0.5),    # Phantom
        (0.3, -0.8, 0.0),    # w0wa with w0 > -1
        (0.2, -1.0, 0.0),    # Low Omega_m
        (0.4, -1.0, 0.0),    # High Omega_m
        (0.3, -1.0, -1.0),   # Strong wa
        (0.3, -1.0, 1.0),    # Strong positive wa
        (0.3, -0.5, 0.0),    # High w0
        (0.3, -1.5, 0.5),    # Very phantom
    ]
    
    a_values = np.linspace(0.05, 0.99, 40)
    
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    best_x = x0.copy()
    best_err = _compute_relative_errors(x0, test_models, a_values, beta=20.0)
    
    # Phase 1: Nelder-Mead with moderate beta
    for i in range(30):
        if time.time() > deadline - 15:
            break
        
        result = minimize(lambda x: _compute_relative_errors(x, test_models, a_values, beta=20.0),
                         best_x, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
        
        if result.fun < best_err:
            best_err = result.fun
            best_x = result.x.copy()
    
    # Phase 2: Polish with higher beta (sharper max)
    if time.time() < deadline - 10:
        result = minimize(lambda x: _compute_relative_errors(x, test_models, a_values, beta=50.0),
                         best_x, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_err:
            best_err = result.fun
            best_x = result.x.copy()
    
    # Clamp to [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    
    # Verify gamma bounds on a fine grid
    gamma_violation = False
    for Om0, w0, wa in test_models:
        for a in np.linspace(0.01, 0.999, 100):
            Om_a = _compute_om_a(a, Om0, w0, wa)
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            gamma = best_x[0] + best_x[1]*q + best_x[2]*q**2 + best_x[3]*u + best_x[4]*wa*a + best_x[5]*u*q
            if gamma < 0.1 or gamma > 1.2:
                gamma_violation = True
                break
        if gamma_violation:
            break
    
    if gamma_violation:
        # Fall back to seed if gamma bounds violated
        best_x = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return list(best_x)
# EVOLVE-BLOCK-END
