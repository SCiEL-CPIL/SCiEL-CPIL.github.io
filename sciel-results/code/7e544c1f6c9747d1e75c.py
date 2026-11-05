"""Improved six-coefficient growth-index approximation for CPL dark energy."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    # Coefficients for gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    
    def compute_reference(om_m, w0, wa, a_vals):
        """
        Solve the ODE for the growth factor D(a) and compute f(a) and D(a)/D(1).
        f(a) = d ln D / d ln a = a * D' / D
        ODE: D'' + (3 + w(a)) H(a) D' - 2 om_m(a) H(a)^2 D = 0
        """
        # We integrate from a=1 down to a_min or up to a_max.
        # It's often more stable to integrate from a=1 (present) to a<1.
        # Let x = ln(a). dx = da/a.
        # D'' + (3+w) H D' - 2 om_m H^2 D = 0
        # Let y1 = D, y2 = D'.
        # y1' = y2
        # y2' = -(3+w) H y2 + 2 om_m H^2 y1
        
        # H(a) = H0 * sqrt(om_m a^-3 + om_de a^-3(1+3w0+3wa) ... )
        # For CPL: rho_de(a) = rho_de0 * a^-3(1+w0+wa) * exp(3wa(1-a))
        # Let's use the standard form:
        # E(a)^2 = om_m a^-3 + (1-om_m) a^-3(1+w0+wa) exp(3wa(1-a))
        # Actually, the standard CPL density is:
        # rho_de(a) = rho_de0 a^{-3(1+w0+wa)} exp(3wa(1-a))
        # So H(a)^2 = H0^2 [ om_m a^-3 + (1-om_m) a^{-3(1+w0+wa)} exp(3wa(1-a)) ]
        
        def ode_func(x, y):
            a = np.exp(x)
            # H(a) in units of H0
            de_exp = np.exp(3 * wa * (1 - a))
            om_de_a = (1 - om_m) * a ** (-3 * (1 + w0 + wa)) * de_exp
            om_m_a = om_m * a ** -3
            H2 = om_m_a + om_de_a
            H = np.sqrt(H2)
            
            w_a_val = w0 + wa * (1 - a)
            
            # y = [D, D']
            D = y[0]
            D_prime = y[1]
            
            # dD/dx = D'
            dD_dx = D_prime
            # dD'/dx = -(3+w)H D' + 2 om_m H^2 D
            dD_prime_dx = -(3 + w_a_val) * H * D_prime + 2 * om_m_a * H2 * D
            
            return [dD_dx, dD_prime_dx]
        
        # Integrate from x=0 (a=1) to x_min = ln(a_min)
        # We need a grid of a values.
        # Let's pick a range a in [0.05, 1.0]
        a_min = 0.05
        a_max = 1.0
        a_grid = np.linspace(a_min, a_max, 50)
        x_grid = np.log(a_grid)
        
        # Initial conditions at a=1 (x=0)
        # D(1) = 1, D'(1) = f(1)
        # f(1) can be estimated analytically or set to 1.0 (linear growth in matter dominated era is f~1, but here it's mixed).
        # Actually, we can solve backwards or forwards.
        # Let's solve from a=1 to a=a_min.
        # y(0) = [1, f(1)]
        # We need a good initial f(1).
        # For flat universe, f(1) ~ 0.55 * om_m^0.15 + 0.1 * (1+w0) * (1-om_m)
        f_init = 0.55 * om_m**0.15 + 0.1 * (1 + w0) * (1 - om_m)
        
        sol = solve_ivp(ode_func, [0, np.log(a_min)], [1.0, f_init], t_eval=x_grid, rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            return None, None
            
        D_vals = sol.y[0]
        D_prime_vals = sol.y[1]
        
        # Normalize D such that D(1)=1 (it is already 1 at x=0)
        # f(a) = a * D' / D
        f_vals = a_grid * D_prime_vals / D_vals
        
        # Normalized growth factor D(a)/D(1)
        # D(1) is sol.y[0][-1] if we integrated backwards? 
        # solve_ivp with t_eval from 0 to negative values returns points in that order.
        # x_grid[0] = ln(a_min), x_grid[-1] = 0.
        # So sol.y[0][-1] is D(1).
        D_norm = D_vals / sol.y[0][-1]
        
        return f_vals, D_norm

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        # Sample cosmologies
        # Om_m in [0.2, 0.4]
        # w0 in [-1.2, -0.8]
        # wa in [-0.5, 0.5]
        om_ms = [0.25, 0.30, 0.35]
        w0s = [-1.0, -0.9, -1.1]
        was = [0.0, 0.3, -0.3]
        
        # Limit number of evaluations for speed, but enough for robustness
        # We'll iterate over a subset
        for om_m in om_ms:
            for w0 in w0s:
                for wa in was:
                    # Skip unphysical if needed, but CPL is general
                    f_ref, D_ref = compute_reference(om_m, w0, wa, None)
                    if f_ref is None:
                        continue
                    
                    a_vals = np.linspace(0.05, 1.0, 20)
                    
                    # Compute analytic values
                    # Om(a) = om_m * a^-3 / (om_m * a^-3 + (1-om_m) * a^(-3(1+w0+wa)) * exp(3wa(1-a)))
                    de_exp = np.exp(3 * wa * (1 - a_vals))
                    om_de_a = (1 - om_m) * a_vals ** (-3 * (1 + w0 + wa)) * de_exp
                    om_m_a = om_m * a_vals ** -3
                    om_a = om_m_a / (om_m_a + om_de_a)
                    
                    q = 1 - om_a
                    u = 1 + w0 + wa * (1 - a_vals)
                    
                    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
                    
                    # Check gamma bounds
                    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                        return 1000.0 # Penalize out of bounds
                    
                    f_ana = om_a ** gamma
                    
                    # Calculate D_ana
                    # D(a)/D(1) = exp(- integral_ln(a)^0 f(exp(x)) dx)
                    # Let's integrate numerically or use cumulative trapz
                    # x = ln(a)
                    x_vals = np.log(a_vals)
                    # f is a function of a, so f(x) = f_ana
                    # We need integral from x to 0 of f(x') dx'
                    # Since a_vals is increasing from 0.05 to 1.0, x_vals is increasing from ln(0.05) to 0.
                    # We want D(a) = exp(- ∫_a^1 f(a')/a' da') = exp(- ∫_ln(a)^0 f(exp(x')) dx')
                    # Let I(a) = ∫_ln(a)^0 f(exp(x')) dx'
                    # We can compute this by integrating from 0 down to ln(a).
                    # Or integrate from ln(a_min) to 0 and subtract.
                    
                    # Simpler: use cumulative integral from a=1 downwards?
                    # Let's just compute the integral for each point using trapz on the grid.
                    # This is O(N^2) which is fine for N=20 and few models.
                    D_ana = np.ones_like(a_vals)
                    for i, a in enumerate(a_vals):
                        if a == 1.0:
                            D_ana[i] = 1.0
                            continue
                        # Mask points where a' >= a
                        mask = a_vals >= a
                        a_int = a_vals[mask]
                        f_int = f_ana[mask]
                        # Integrate f(a')/a' da' from a to 1
                        # Note: f(a') is defined as Om(a')^gamma.
                        # The integral in the task is ∫ f(exp(x)) dx.
                        # dx = da/a.
                        # So ∫_a^1 f(a')/a' da'.
                        integrand = f_int / a_int
                        val = np.trapz(integrand, a_int)
                        D_ana[i] = np.exp(-val)
                    
                    # Relative errors
                    err_f = np.max(np.abs((f_ana - f_ref) / f_ref))
                    err_D = np.max(np.abs((D_ana - D_ref) / D_ref))
                    
                    curr_max = max(err_f, err_D)
                    if curr_max > max_err:
                        max_err = curr_max
                    
                    # Early break if error is already large
                    if max_err > 0.5:
                        return max_err
                        
        return max_err

    # Initial guess based on Linder
    x0 = [0.55, -0.05, 0.01, 0.1, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead or Powell for robustness with non-smooth max error
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5})
    
    best_coeffs = res.x
    # Ensure finite and within bounds
    if not np.all(np.isfinite(best_coeffs)):
        return x0
        
    # Clip to [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
