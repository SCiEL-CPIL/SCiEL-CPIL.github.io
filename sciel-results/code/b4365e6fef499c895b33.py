"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def _om_a(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    if a <= 0:
        return 1.0
    a = max(a, 1e-15)
    de_exp = -3.0 * (1.0 + w0 + wa * (1.0 - a))
    if a >= 1.0:
        de_factor = 1.0
    else:
        one_minus_a = max(1.0 - a, 1e-15)
        de_factor = one_minus_a ** de_exp
    E2 = Om0 / (a ** 3) + (1.0 - Om0) * de_factor
    return (Om0 / (a ** 3)) / E2

def _solve_growth_ode(Om0, w0, wa, a_min=0.05, n_points=50):
    """
    Solve the GR growth ODE for f = dlnD/dlna.
    f' + f^2 + (1/2)(3*Om(a) - 1) = 0  [with respect to ln a]
    Initial condition: f(a=1) ~ 0.55 * Om0^0.55 (standard approximation)
    """
    a_grid = np.linspace(a_min, 1.0, n_points)
    x_grid = np.log(a_grid)  # x = ln a
    
    def f_prime(x, f):
        a = np.exp(x)
        Om = _om_a(a, Om0, w0, wa)
        return -f[0]**2 - 0.5 * (3.0 * Om - 1.0)
    
    # Initial condition at x=0 (a=1)
    f0 = 0.55 * Om0 ** 0.55
    sol = solve_ivp(f_prime, [0.0, np.log(a_min)], [f0], t_eval=x_grid, 
                    method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        # Fallback to power-law approximation
        a_vals = np.exp(sol.t)
        Om_vals = np.array([_om_a(a, Om0, w0, wa) for a in a_vals])
        f_vals = Om_vals ** 0.55
        return a_vals, f_vals, Om_vals
    
    return np.exp(sol.t), sol.y[0], np.array([_om_a(a, Om0, w0, wa) for a in np.exp(sol.t)])

def _gamma_of_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _objective(coeffs):
    """Compute worst relative error across test cosmologies for both f and D."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Test cosmologies: (Om0, w0, wa)
    test_models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, -0.3),
        (0.3, -1.2, 0.5),
        (0.3, -0.8, 0.0),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.0, -1.0),
        (0.3, -1.0, 1.0),
        (0.3, -0.5, 0.0),
        (0.3, -1.5, 0.5),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.2),
    ]
    
    a_values = np.linspace(0.05, 0.99, 25)
    
    max_rel_err = 0.0
    
    for Om0, w0, wa in test_models:
        # Get reference solution from ODE
        a_ref, f_ref_arr, Om_ref_arr = _solve_growth_ode(Om0, w0, wa, a_min=0.05, n_points=25)
        
        for i, a in enumerate(a_values):
            if a <= 0:
                continue
            
            Om = _om_a(a, Om0, w0, wa)
            q = 1.0 - Om
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = _gamma_of_coeffs(coeffs, q, u, wa, a)
            
            if gamma < 0.1 or gamma > 1.2:
                return 10000.0
            
            f_approx = Om ** gamma
            
            # Find closest reference point
            idx = np.argmin(np.abs(a_ref - a))
            f_ref = f_ref_arr[idx]
            
            if abs(f_ref) < 1e-10:
                continue
            
            rel_err_f = abs(f_approx - f_ref) / max(abs(f_ref), 1e-10)
            max_rel_err = max(max_rel_err, rel_err_f)
    
    return max_rel_err * 100

def construct():
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    best_x = np.array(x0, dtype=float)
    best_err = _objective(best_x)
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 8
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            x_start = best_x.copy()
        else:
            # Physics-constrained initialization: small perturbations around known-good values
            x_start = np.array([
                0.55 + rng.uniform(-0.1, 0.1),  # c0
                rng.uniform(-0.5, 0.5),          # c1
                rng.uniform(-0.5, 0.5),          # c2
                rng.uniform(-0.5, 0.5),          # c3
                rng.uniform(-0.5, 0.5),          # c4
                rng.uniform(-0.5, 0.5),          # c5
            ])
        
        try:
            result = minimize(_objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, best_x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x
        except Exception:
            pass
    
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)
# EVOLVE-BLOCK-END
