"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Return six coefficients [c0..c5] for the growth-index ansatz:
      q = 1 - Om(a)
      u = 1 + w0 + wa*(1-a)
      gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
      f = Om(a)^gamma
    """
    t_start = time.time()
    deadline = t_start + 80.0  # 10s safety margin before 90s kill

    # Best known: constant gamma ~ 0.55
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = float('inf')

    # --- Helper functions ---
    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        # Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*exp(3*(1+w0+wa/2)*(1-a)))
        # More precisely: E^2(a) = Om0*a^-3 + (1-Om0)*exp(-3*int_1^a (1+w(a'))/a' da')
        # For CPL: w(a') = w0 + wa*(1-a'), so int = (1+w0+wa/2)*(1-a)
        # Thus E^2(a) = Om0*a^-3 + (1-Om0)*exp(3*(1+w0+wa/2)*(a-1))
        # Wait, let me re-derive:
        # rho_DE/rho_DE(1) = exp(3*int_1^a (1+w(a'))/a' da')
        # int_1^a (1+w0+wa*(1-a'))/a' da' = int_1^a (1+w0)/a' da' + wa*int_1^a (1-a')/a' da'
        # = (1+w0)*ln(a) + wa*int_1^a (1/a' - 1) da'
        # = (1+w0)*ln(a) + wa*(ln(a) - (a-1))
        # = (1+w0+wa)*ln(a) - wa*(a-1)
        # So rho_DE/rho_DE(1) = exp(3*((1+w0+wa)*ln(a) - wa*(a-1)))
        # = a^(3*(1+w0+wa)) * exp(-3*wa*(a-1))
        # E^2(a) = Om0/a^3 + (1-Om0)*a^(3*(1+w0+wa))*exp(-3*wa*(a-1))
        # Om(a) = (Om0/a^3) / E^2(a)
        
        a = np.asarray(a, dtype=float)
        if np.isscalar(a) or a.ndim == 0:
            a = np.array([a])
        
        Om_m = Om0 * a**(-3)
        # DE term
        if abs(wa) < 1e-15:
            Om_de = (1.0 - Om0) * a**(3.0*(1.0 + w0))
        else:
            Om_de = (1.0 - Om0) * a**(3.0*(1.0 + w0 + wa)) * np.exp(-3.0*wa*(a - 1.0))
        
        E2 = Om_m + Om_de
        Om_a = Om_m / E2
        return Om_a

    def gamma_from_coeffs(coeffs, a, Om0, w0, wa):
        """Compute gamma from the ansatz."""
        c0, c1, c2, c3, c4, c5 = coeffs
        Om_a = Om_of_a(a, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        return gamma

    def f_approx(coeffs, a, Om0, w0, wa):
        """Compute f = Om(a)^gamma."""
        Om_a = Om_of_a(a, Om0, w0, wa)
        gamma = gamma_from_coeffs(coeffs, a, Om0, w0, wa)
        # Clip gamma to [0.1, 1.2] to avoid issues
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_a**gamma_clipped

    def D_normalized_approx(coeffs, a, Om0, w0, wa):
        """
        Compute D(a)/D(1) = exp(-integral from a to 1 of f(exp(x)) dx)
        where x = ln(a'), so a' = exp(x), da' = exp(x) dx
        integral_ln(a)^0 f(exp(x)) dx = integral_a^1 f(a')/a' da'
        """
        # We need to compute: -int_a^1 f(a')/a' da'
        # Using numerical quadrature
        a_arr = np.asarray(a, dtype=float)
        if a_arr.ndim == 0:
            a_arr = np.array([a_arr])
        
        D_vals = np.ones_like(a_arr)
        for i, ai in enumerate(a_arr):
            if ai >= 1.0:
                D_vals[i] = 1.0
                continue
            # Integrate from ai to 1
            def integrand(log_x):
                a_prime = np.exp(log_x)
                f_val = f_approx(coeffs, a_prime, Om0, w0, wa)
                return f_val / a_prime
            
            # Use fixed quadrature
            n_pts = 200
            log_x_vals = np.linspace(math.log(ai), 0.0, n_pts)
            vals = np.array([integrand(lx) for lx in log_x_vals])
            integral = np.trapz(vals, log_x_vals)
            D_vals[i] = np.exp(-integral)
        
        if len(D_vals) == 1:
            return D_vals[0]
        return D_vals

    def growth_ode_reference(a_grid, Om0, w0, wa):
        """
        Solve the GR growth ODE for f(a) = dlnD/dlna:
        f' + [3/2 + H'/H] f = (3/2) Om(a)
        where H'/H = dH/dlna / H
        """
        # H^2 = H0^2 * E^2(a), so H'/H = dlnE^2/dlna / 2
        # dlnE^2/dlna = a * dE^2/da / E^2
        # E^2 = Om0/a^3 + (1-Om0)*a^(3(1+w0+wa))*exp(-3wa(a-1))
        
        def E2(a):
            Om_m = Om0 * a**(-3)
            if abs(wa) < 1e-15:
                Om_de = (1.0 - Om0) * a**(3.0*(1.0 + w0))
            else:
                Om_de = (1.0 - Om0) * a**(3.0*(1.0 + w0 + wa)) * np.exp(-3.0*wa*(a - 1.0))
            return Om_m + Om_de
        
        def dE2_da(a):
            dOm_m = -3.0 * Om0 * a**(-4)
            if abs(wa) < 1e-15:
                dOm_de = (1.0 - Om0) * 3.0*(1.0 + w0) * a**(3.0*(1.0 + w0) - 1)
            else:
                base = (1.0 - Om0) * a**(3.0*(1.0 + w0 + wa)) * np.exp(-3.0*wa*(a - 1.0))
                dOm_de = base * (3.0*(1.0 + w0 + wa)/a - 3.0*wa)
            return dOm_m + dOm_de
        
        def H_prime_over_H(a):
            e2 = E2(a)
            de2 = dE2_da(a)
            return 0.5 * a * de2 / e2
        
        def Om_a(a):
            e2 = E2(a)
            return Om0 * a**(-3) / e2
        
        def ode_rhs(lna, y):
            a = np.exp(lna)
            f = y[0]
            HpH = H_prime_over_H(a)
            Om = Om_a(a)
            df_dlna = -(3.0/2.0 + HpH) * f + (3.0/2.0) * Om
            return [df_dlna]
        
        # Integrate from a=1 (lna=0) backwards to a_grid[0]
        lna_grid = np.log(a_grid)
        # Solve from lna=0 to lna_grid[0]
        # Initial condition: f(1) ~ Om(1)^0.55 as starting point, but better to use
        # the known asymptotic: for small perturbation, f ~ Om^0.55
        # Actually, we can integrate forward from a small value to a=1, or backward from a=1
        # Let's integrate backward from lna=0
        lna_min = lna_grid[0]
        t_span = (0.0, lna_min)
        t_eval = lna_grid
        y0 = [Om0**0.55]  # f(1) = Om(1)^0.55 = 1^0.55 = 1? No, Om(1)=Om0
        # Actually at a=1, Om(1) = Om0, so f(1) = Om0^gamma
        # But for the ODE, we need a consistent initial condition.
        # Let's use f(1) = Om0^0.55 as a reasonable start
        y0 = [Om0**0.55]
        
        sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, method='RK45', 
                       rtol=1e-10, atol=1e-12, max_step=0.01)
        
        if not sol.success:
            return None
        
        f_ref = sol.y[0]
        
        # Compute D from f: D(a)/D(1) = exp(-int_a^1 f(a')/a' da')
        # = exp(-int_{lna}^0 f(exp(x)) dx)
        D_ref = np.zeros_like(a_grid)
        D_ref[-1] = 1.0  # at a=1
        # Cumulative integral from the end
        for i in range(len(a_grid) - 2, -1, -1):
            # Integrate from lna[i] to lna[i+1]
            n_sub = 50
            x_vals = np.linspace(lna_grid[i], lna_grid[i+1], n_sub)
            y_vals = np.interp(x_vals, lna_grid, f_ref)
            integrand = y_vals / np.exp(x_vals)
            integral = np.trapz(integrand, x_vals)
            D_ref[i] = D_ref[i+1] * np.exp(-integral)
        
        return f_ref, D_ref

    def compute_score(coeffs, cosmologies, a_grid):
        """Compute the max relative error across all cosmologies and a values."""
        max_err_f = 0.0
        max_err_D = 0.0
        
        for (Om0, w0, wa) in cosmologies:
            # Get reference
            ref = growth_ode_reference(a_grid, Om0, w0, wa)
            if ref is None:
                continue
            f_ref, D_ref = ref
            
            # Approximate
            f_app = f_approx(coeffs, a_grid, Om0, w0, wa)
            D_app = D_normalized_approx(coeffs, a_grid, Om0, w0, wa)
            
            # Relative errors (avoid division by zero)
            mask_f = np.abs(f_ref) > 1e-10
            if np.any(mask_f):
                rel_err_f = np.abs((f_app[mask_f] - f_ref[mask_f]) / f_ref[mask_f])
                max_err_f = max(max_err_f, np.max(rel_err_f))
            
            mask_D = np.abs(D_ref) > 1e-10
            if np.any(mask_D):
                rel_err_D = np.abs((D_app[mask_D] - D_ref[mask_D]) / D_ref[mask_D])
                max_err_D = max(max_err_D, np.max(rel_err_D))
        
        return 100.0 * max(max_err_f, max_err_D)

    # --- Define validation cosmologies (plausible CPL models) ---
    # Om0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    cosmologies = [
        (0.3, -1.0, 0.0),    # LCDM-like
        (0.3, -1.0, 0.5),    # CPL with positive wa
        (0.3, -1.0, -0.5),   # CPL with negative wa
        (0.3, -0.8, 0.0),    # w0 = -0.8
        (0.3, -1.2, 0.0),    # w0 = -1.2
        (0.25, -1.0, 0.3),
        (0.35, -1.0, -0.3),
        (0.3, -0.9, 0.2),
        (0.3, -1.1, 0.4),
        (0.3, -1.0, 0.8),
        (0.3, -1.0, -0.8),
        (0.28, -1.05, 0.1),
        (0.32, -0.95, -0.1),
    ]
    
    a_grid = np.linspace(0.1, 1.0, 50)

    # --- Objective function for optimization ---
    def objective(coeffs):
        if time.time() > deadline:
            return best_score
        # Check gamma constraint
        for (Om0, w0, wa) in cosmologies:
            gamma = gamma_from_coeffs(coeffs, a_grid, Om0, w0, wa)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
        # Check coefficient bounds
        if np.any(np.abs(coeffs) > 10.0):
            return 1e10
        return compute_score(coeffs, cosmologies, a_grid)

    # --- Initial optimization ---
    x0 = np.array(best_coeffs)
    bounds = [(-10, 10)] * 6
    
    # Try Nelder-Mead first
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                               'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = list(res.x)
    except Exception:
        pass

    if time.time() > deadline:
        return best_coeffs

    # Try L-BFGS-B with the smooth surrogate
    def objective_smooth(coeffs):
        if time.time() > deadline:
            return best_score
        for (Om0, w0, wa) in cosmologies:
            gamma = gamma_from_coeffs(coeffs, a_grid, Om0, w0, wa)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
        if np.any(np.abs(coeffs) > 10.0):
            return 1e10
        
        # Use log-sum-exp for smooth max
        max_err_f = 0.0
        max_err_D = 0.0
        err_list_f = []
        err_list_D = []
        
        for (Om0, w0, wa) in cosmologies:
            ref = growth_ode_reference(a_grid, Om0, w0, wa)
            if ref is None:
                continue
            f_ref, D_ref = ref
            f_app = f_approx(coeffs, a_grid, Om0, w0, wa)
            D_app = D_normalized_approx(coeffs, a_grid, Om0, w0, wa)
            
            mask_f = np.abs(f_ref) > 1e-10
            if np.any(mask_f):
                rel_err_f = np.abs((f_app[mask_f] - f_ref[mask_f]) / f_ref[mask_f])
                err_list_f.extend(rel_err_f.tolist())
            
            mask_D = np.abs(D_ref) > 1e-10
            if np.any(mask_D):
                rel_err_D = np.abs((D_app[mask_D] - D_ref[mask_D]) / D_ref[mask_D])
                err_list_D.extend(rel_err_D.tolist())
        
        if not err_list_f or not err_list_D:
            return 1e10
        
        # Log-sum-exp for smooth max
        def logsumexp(arr, beta=50):
            arr = np.array(arr)
            m = np.max(arr)
            return m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta
        
        smooth_f = logsumexp(err_list_f)
        smooth_D = logsumexp(err_list_D)
        return 100.0 * max(smooth_f, smooth_D)

    try:
        res2 = minimize(objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res2.fun < best_score:
            best_score = res2.fun
            best_coeffs = list(res2.x)
    except Exception:
        pass

    if time.time() > deadline:
        return best_coeffs

    # Final polish with tighter Nelder-Mead
    try:
        res3 = minimize(objective, np.array(best_coeffs), method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if res3.fun < best_score:
            best_score = res3.fun
            best_coeffs = list(res3.x)
    except Exception:
        pass

    # Ensure coefficients are in bounds
    best_coeffs = [max(-10.0, min(10.0, c)) for c in best_coeffs]
    
    return best_coeffs

# EVOLVE-BLOCK-END
