import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Helper functions for CPL cosmology
    om0 = 0.3
    
    def get_e2(a, w0, wa):
        """Calculate E(a)^2 for CPL dark energy."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = np.array([a])
            
        x = 3 * wa * (1 - a)
        rho_de = a**(-3 * (1 + w0 + wa)) * np.exp(x)
        rho_m = a**(-3)
        e2 = om0 * rho_m + (1 - om0) * rho_de
        
        if scalar_input:
            return e2[0]
        return e2

    def get_om_m(a, w0, wa):
        """Calculate Omega_m(a)."""
        a = np.asarray(a, dtype=float)
        scalar_input = a.ndim == 0
        if scalar_input:
            a = np.array([a])
            
        e2_val = get_e2(a, w0, wa)
        rho_m = a**(-3)
        om_m = om0 * rho_m / e2_val
        
        if scalar_input:
            return om_m[0]
        return om_m

    # 1. Define Validation Cosmologies
    # Use a broader range of CPL parameters for better generalization
    w0_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    wa_vals = [-0.5, -0.25, 0.0, 0.25, 0.5]
    
    # Scale factors for evaluation - use logarithmic spacing for better coverage
    a_grid = np.logspace(np.log10(0.05), 0.0, 30)

    # Pre-compute reference data
    ref_data = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            if time.time() - start_time > deadline - 5:
                break
                
            def ode_f(t, y):
                a = np.exp(t)
                f = y[0]
                om = get_om_m(a, w0, wa)
                ol = 1 - om
                w = w0 + wa * (1 - a)
                h_prime_h = 0.5 * om - 0.5 * ol * (1 + 3 * w)
                return [f**2 - (2 + h_prime_h) * f - 1.5 * om]

            # Integrate from a=0.01 to a=1
            t_span = (np.log(0.01), 0.0)
            try:
                sol = solve_ivp(ode_f, t_span, [1.0], t_eval=np.log(a_grid), method='RK45', rtol=1e-8, atol=1e-10)
            except Exception:
                continue
            
            if not sol.success:
                continue

            f_exact = sol.y[0]
            
            # Calculate D_exact using trapezoidal integration
            dC = np.zeros_like(sol.t)
            dC[1:] = 0.5 * (sol.y[0,1:] + sol.y[0,:-1]) * (sol.t[1:] - sol.t[:-1])
            C = np.cumsum(dC)
            
            D_exact = np.zeros_like(a_grid)
            for i in range(len(a_grid)):
                integral_val = C[-1] - C[i]
                D_exact[i] = np.exp(-integral_val)
                
            ref_data.append((w0, wa, a_grid, f_exact, D_exact))

    if not ref_data:
        # Fallback to Linder (2005) approximation: gamma = 0.55
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 2. Define Approximation
    def approx_growth(c, w0, wa, a):
        c0, c1, c2, c3, c4, c5 = c
        
        a = np.asarray(a, dtype=float)
        om_m = get_om_m(a, w0, wa)
        q = 1 - om_m
        u = 1 + w0 + wa * (1 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        # Clip gamma to valid range for calculation
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        
        f_app = om_m ** gamma_clipped
        
        # D_app calculation using vectorized approach
        D_app = np.zeros_like(a)
        
        # Use a fine grid for integration
        a_fine = np.linspace(0.05, 1.0, 200)
        om_fine = get_om_m(a_fine, w0, wa)
        q_fine = 1 - om_fine
        u_fine = 1 + w0 + wa * (1 - a_fine)
        
        gamma_fine = c0 + c1 * q_fine + c2 * q_fine**2 + c3 * u_fine + c4 * wa * a_fine + c5 * u_fine * q_fine
        gamma_fine_clipped = np.clip(gamma_fine, 0.1, 1.2)
        f_fine = om_fine ** gamma_fine_clipped
        
        # Precompute integral F(a) = integral_a^1 f(a')/a' da'
        # Using cumulative trapezoidal rule
        integrand = f_fine / a_fine
        dF = np.zeros_like(a_fine)
        dF[1:] = 0.5 * (integrand[1:] + integrand[:-1]) * (a_fine[1:] - a_fine[:-1])
        F_cum = np.cumsum(dF)
        F_total = F_cum[-1]
        
        # For each a in a, find integral from a to 1
        for i, av in enumerate(a):
            if av >= 1.0:
                D_app[i] = 1.0
                continue
            
            # Find the index closest to av
            idx = np.searchsorted(a_fine, av)
            if idx == 0:
                integral_val = F_total
            elif idx == len(a_fine):
                integral_val = 0.0
            else:
                # Linear interpolation
                a0, a1 = a_fine[idx-1], a_fine[idx]
                F0, F1 = F_cum[idx-1], F_cum[idx]
                frac = (av - a0) / (a1 - a0)
                F_at_av = F0 + frac * (F1 - F0)
                integral_val = F_total - F_at_av
            
            D_app[i] = np.exp(-integral_val)
        
        return f_app, D_app, gamma

    def objective(c):
        if time.time() - start_time > deadline:
            return 1000.0
            
        max_err = 0.0
        
        for w0, wa, a, f_e, D_e in ref_data:
            f_app, D_app, gamma = approx_growth(c, w0, wa, a)
            
            # Check gamma constraints
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1000.0
            
            # Relative errors
            err_f = np.abs(f_app / f_e - 1.0)
            err_D = np.abs(D_app / D_e - 1.0)
            
            local_max = max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max
                
        return max_err * 100.0

    # Initial guess based on Linder (2005) and physical considerations
    # gamma ≈ 0.55 + small corrections
    x0 = [0.55, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize using multiple strategies
    best_x = x0
    best_val = objective(x0)
    
    # Strategy 1: Nelder-Mead
    if time.time() - start_time < deadline - 15:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass
    
    # Strategy 2: Powell
    if time.time() - start_time < deadline - 10:
        try:
            res2 = minimize(objective, best_x, method='Powell', 
                           options={'maxiter': 150, 'xrtol': 1e-6, 'ftol': 1e-6})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    # Strategy 3: COBYLA for constrained optimization
    if time.time() - start_time < deadline - 5:
        try:
            # Define constraints for gamma range
            def constraint_gamma_min(c):
                _, _, gamma = approx_growth(c, -1.0, -0.5, np.array([0.1]))
                return gamma[0] - 0.1
            
            def constraint_gamma_max(c):
                _, _, gamma = approx_growth(c, 0.5, 0.5, np.array([0.1]))
                return 1.2 - gamma[0]
            
            constraints = [
                {'type': 'ineq', 'fun': constraint_gamma_min},
                {'type': 'ineq', 'fun': constraint_gamma_max}
            ]
            
            res3 = minimize(objective, best_x, method='COBYLA', 
                           constraints=constraints,
                           options={'maxiter': 200, 'rhobeg': 0.1})
            if res3.fun < best_val:
                best_x = res3.x
                best_val = res3.fun
        except Exception:
            pass
    
    # Final refinement with smaller steps
    if time.time() - start_time < deadline - 2:
        try:
            res4 = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-7})
            if res4.fun < best_val:
                best_x = res4.x
                best_val = res4.fun
        except Exception:
            pass
    
    # Clamp coefficients to [-10, 10]
    best_c = [np.clip(val, -10, 10) for val in best_x]
    
    # Validate final result
    try:
        final_err = objective(best_c)
        if final_err > 1000:  # Invalid
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return best_c

# EVOLVE-BLOCK-END
