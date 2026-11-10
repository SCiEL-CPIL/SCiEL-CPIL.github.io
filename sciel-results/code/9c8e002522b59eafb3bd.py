"""Growth-index coefficient optimizer for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six finite coefficients in [-10,10] for the growth-index ansatz."""
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    # Helper functions for the ansatz
    def _Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        a3 = a**3
        de = np.exp(3.0 * wa * (1.0 - a))
        denom = a3 + de
        return Om0 * a3 / denom
    
    def _q_of_a(a, Om0, w0, wa):
        """Deceleration parameter q = 1 - Om(a)."""
        return 1.0 - _Om_of_a(a, Om0, w0, wa)
    
    def _u_of_a(a, w0, wa):
        """u = 1 + w0 + wa*(1-a)."""
        return 1.0 + w0 + wa * (1.0 - a)
    
    def _gamma(a, coeffs, Om0, w0, wa):
        """Compute gamma from the six-coefficient ansatz."""
        c0, c1, c2, c3, c4, c5 = coeffs
        q = _q_of_a(a, Om0, w0, wa)
        u = _u_of_a(a, w0, wa)
        g = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        return g
    
    def _f_of_a(a, coeffs, Om0, w0, wa):
        """Compute f = Om(a)^gamma."""
        Om = _Om_of_a(a, Om0, w0, wa)
        g = _gamma(a, coeffs, Om0, w0, wa)
        if Om <= 0:
            return 0.0
        return Om**g
    
    def _D_ratio(a, coeffs, Om0, w0, wa):
        """Compute D(a)/D(1) = exp(-int_a^1 f(e^x) dx)."""
        def integrand(x):
            aa = np.exp(x)
            return _f_of_a(aa, coeffs, Om0, w0, wa)
        if a >= 1.0:
            return 1.0
        val, _ = quad(integrand, np.log(a), 0.0, limit=200)
        return np.exp(-val)
    
    def _reference_growth_ODE(a_arr, Om0, w0, wa):
        """Solve the GR growth ODE to get reference f and D."""
        # ODE for D: D'' + 2H D' - 4πG ρ_m D = 0
        # In terms of ln a: D''(ln a) + (3/2 - q) D'(ln a) - (3/2) Ω_m D = 0
        # Let F = D'/D (growth rate), then F' + F^2 + F(3/2 - q) - (3/2)Ω_m = 0
        
        def ode_rhs(x, y):
            # y = [D, F] where F = D'/D
            D, F = y
            aa = np.exp(x)
            Om = _Om_of_a(aa, Om0, w0, wa)
            q = _q_of_a(aa, Om0, w0, wa)
            dD = F * D
            dF = -(F**2 + F*(1.5 - q) - 1.5*Om)
            return [dD, dF]
        
        # Integrate from x=0 (a=1) backwards to x=ln(a_min)
        # Use D(1)=1, F(1)≈Ω_m(1)^0.55 as initial condition
        F1 = _Om_of_a(1.0, Om0, w0, wa)**0.55
        x0 = 0.0
        x_end = -5.0  # a ~ 0.0067
        
        sol = solve_ivp(ode_rhs, [x0, x_end], [1.0, F1], 
                       method='RK45', rtol=1e-10, atol=1e-12,
                       max_step=0.01, dense_output=True)
        
        if not sol.success:
            return None, None
        
        # Extract f and D at the grid points
        f_ref = np.zeros_like(a_arr)
        D_ref = np.zeros_like(a_arr)
        for i, a in enumerate(a_arr):
            if a >= 1.0:
                f_ref[i] = F1
                D_ref[i] = 1.0
            else:
                x = np.log(a)
                if x < sol.t[0]:
                    # Extrapolate or use nearest
                    idx = 0
                elif x > sol.t[-1]:
                    idx = -1
                else:
                    idx = np.searchsorted(sol.t, x, side='right') - 1
                    idx = max(0, min(idx, len(sol.t)-2))
                D_val, F_val = sol.sol(x)
                f_ref[i] = F_val
                D_ref[i] = D_val / D_val  # normalized later
        
        # Normalize D so D(1)=1
        D_at_1 = sol.sol(0.0)[0]
        D_ref = D_ref / D_at_1 if D_at_1 != 0 else D_ref
        
        return f_ref, D_ref
    
    # Define validation cosmologies (CPL models)
    # Om0 in [0.2, 0.4], w0 in [-1.2, -0.5], wa in [-0.5, 0.5]
    validation_models = [
        (0.30, -1.0, 0.0),   # ΛCDM
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.9, 0.1),
        (0.30, -1.1, -0.1),
        (0.25, -0.8, 0.2),
        (0.35, -1.2, -0.2),
        (0.30, -0.7, 0.3),
        (0.28, -1.05, 0.05),
        (0.32, -0.95, -0.05),
    ]
    
    # Grid of a values for evaluation
    a_grid = np.geomspace(0.01, 1.0, 50)
    
    # Precompute reference solutions
    refs = []
    for Om0, w0, wa in validation_models:
        f_ref, D_ref = _reference_growth_ODE(a_grid, Om0, w0, wa)
        if f_ref is not None:
            refs.append((Om0, w0, wa, f_ref, D_ref))
    
    if len(refs) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _objective(coeffs):
        """Compute max relative error across all validation models and a values."""
        max_err = 0.0
        for Om0, w0, wa, f_ref, D_ref in refs:
            for i, a in enumerate(a_grid):
                # Check gamma constraint
                g = _gamma(a, coeffs, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    return 1e10  # Penalty for constraint violation
                
                # Compute f and D for this model
                f_val = _f_of_a(a, coeffs, Om0, w0, wa)
                D_val = _D_ratio(a, coeffs, Om0, w0, wa)
                
                # Relative errors
                if abs(f_ref[i]) > 1e-10:
                    err_f = abs(f_val - f_ref[i]) / abs(f_ref[i])
                else:
                    err_f = abs(f_val - f_ref[i])
                
                if abs(D_ref[i]) > 1e-10:
                    err_D = abs(D_val - D_ref[i]) / abs(D_ref[i])
                else:
                    err_D = abs(D_val - D_ref[i])
                
                err = max(err_f, err_D)
                if err > max_err:
                    max_err = err
        
        return max_err
    
    # Initial guess: constant gamma = 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Use differential evolution for global search, then polish
    best_result = None
    best_score = _objective(x0)
    
    # Quick local refinement with Nelder-Mead
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        # Not enough time for DE, return best so far
        if best_result is not None:
            # Clip to [-10, 10]
            best_result = np.clip(best_result, -10, 10)
            return list(best_result)
        return x0
    
    # Differential evolution for broader search
    bounds = [(-10, 10)] * 6
    try:
        de_result = differential_evolution(_objective, bounds, 
                                          seed=42, maxiter=50, 
                                          tol=1e-8, polish=True,
                                          mutation=(0.5, 1.0), recombination=0.7,
                                          popsize=15, updating='deferred')
        if de_result.success and de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        if best_result is not None:
            best_result = np.clip(best_result, -10, 10)
            return list(best_result)
        return x0
    
    # Final high-precision polish
    if best_result is not None:
        try:
            polish_res = minimize(_objective, best_result, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            if polish_res.success and polish_res.fun < best_score:
                best_score = polish_res.fun
                best_result = polish_res.x
        except Exception:
            pass
    
    if best_result is not None:
        best_result = np.clip(best_result, -10, 10)
        return list(best_result)
    return x0
# EVOLVE-BLOCK-END
