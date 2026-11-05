import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s safety margin
    
    # 1. Define the reference growth calculation
    def get_reference_growth(Om0, w0, wa, a_vals):
        # a_vals: array of scale factors
        # Returns f_ref, D_ref_normalized
        OmL0 = 1.0 - Om0
        
        def omega_m(a):
            # CPL density ratio
            # rho_L(a) / rho_L(1) = a^{-3(1+w0+wa)} * exp(3*wa*a)
            # rho_m(a) / rho_m(1) = a^{-3}
            term_L = (a ** (-3.0 * (1.0 + w0 + wa))) * np.exp(3.0 * wa * a)
            term_m = a ** (-3.0)
            denom = Om0 * term_m + OmL0 * term_L
            return (Om0 * term_m) / denom
        
        def omega_m_prime(a):
            # Numerical derivative for simplicity and robustness
            h = 1e-5
            return (omega_m(a + h) - omega_m(a - h)) / (2.0 * h)
        
        def growth_ode(a, y):
            # y = [D, D']
            D, Dp = y
            om = omega_m(a)
            om_p = omega_m_prime(a)
            
            # H'/H term
            # H^2 prop to Om + OmL
            # d(H^2)/da = 2 H H' => H'/H = 0.5 * (d/d a (H^2)) / H^2
            # H^2 prop to (Om0 * a^-3 + OmL0 * a^-3(1+w0+wa) * exp(3wa a))
            # Let F(a) = Om0 a^-3 + OmL0 a^-3(1+w0+wa) exp(3wa a)
            # F'(a) = -3 Om0 a^-4 + OmL0 [ -3(1+w0+wa) a^-4(1+w0+wa) exp(3wa a) + 3wa a^-3(1+w0+wa) exp(3wa a) ]
            # This is getting messy. Let's use numerical derivative for H2 as well.
            
            def H2(a):
                term_L = (a ** (-3.0 * (1.0 + w0 + wa))) * np.exp(3.0 * wa * a)
                term_m = a ** (-3.0)
                return Om0 * term_m + OmL0 * term_L
            
            H2_val = H2(a)
            H2_p = (H2(a + h) - H2(a - h)) / (2.0 * h)
            H_over_H_prime = 0.5 * H2_p / H2_val
            
            coeff1 = 3.0 / a + H_over_H_prime
            coeff2 = 1.5 * om / (a * a)
            
            return [Dp, -coeff1 * Dp + coeff2 * D]
        
        # Integrate from a=1 downwards or upwards?
        # Easier to integrate from a_small to a=1 and normalize.
        # Or integrate from 1 to a.
        # Let's integrate from a_min to 1.
        a_min = np.min(a_vals)
        a_max = 1.0
        
        # Initial conditions at a_min (approximation)
        # f ~ Om^0.55 is good at high z. D ~ a^1.5?
        # Let's start at a_min with D=1, Dp=1.5 * a_min**0.5? No, D ~ a^1.5 for matter dominated.
        # D'(a) = 1.5 a^0.5.
        D_init = 1.0
        Dp_init = 1.5 * np.sqrt(a_min)
        
        sol = solve_ivp(growth_ode, [a_min, a_max], [D_init, Dp_init], 
                        t_eval=a_vals, method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            return None, None
        
        D = sol.y[0]
        Dp = sol.y[1]
        
        # Normalize D(1) = 1
        D_norm = D / D[-1]
        
        # f = D' / (a H) ? No, f = d ln D / d ln a = a D' / D
        f_ref = a_vals * Dp / D
        
        return f_ref, D_norm

    # 2. Define the approximation
    def get_approx(Om0, w0, wa, c, a_vals):
        Om_a = get_omega_m(Om0, w0, wa, a_vals)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return None, None, False
            
        f_app = Om_a ** gamma
        
        # Calculate D_app
        # D(a)/D(1) = exp(- \int_a^1 f(exp(x)) dx) ?
        # Prompt: D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)
        # Let x = ln a'. Integral from ln(a) to 0.
        # Let t = ln a'. a' = exp(t). da' = exp(t) dt.
        # Wait, the integral is \int_{\ln a}^0 f(\exp(x)) dx.
        # This is \int_{\ln a}^0 f(a') dx.
        # We need to compute this integral numerically.
        
        # We can use cumulative trapezoidal integration on the grid of x.
        # x_vals = np.log(a_vals)
        # But a_vals might not be sorted in x linearly? They are.
        # We need f at specific x points.
        
        # To be efficient, we can precompute the integral for the grid.
        # However, c changes, so f changes.
        
        # Let's use simple trapezoidal rule on the provided a_vals grid.
        # x = log(a)
        x_vals = np.log(a_vals)
        # Sort by x (increasing a)
        # a_vals is likely increasing.
        
        # Integral I(a) = \int_{\ln a}^0 f(\exp(x)) dx
        # Let J(x) = \int_x^0 f(\exp(t)) dt
        # J(0) = 0
        # J(x) = - \int_0^x f(\exp(t)) dt
        
        # We can compute cumulative integral from x=0 backwards.
        # Or forwards.
        # Let's compute K(x) = \int_0^x f(\exp(t)) dt.
        # Then J(x) = -K(x).
        
        # Use trapezoidal rule on x_vals.
        # We need f at x_vals.
        f_vals = f_app
        
        # np.trapz or cumulative_trapezoid
        # scipy.integrate.cumulative_trapezoid
        from scipy.integrate import cumulative_trapezoid
        
        # x_vals must be sorted.
        # a_vals is increasing, so x_vals is increasing.
        # We want integral from 0 to x.
        # But our grid starts at a_min (x_min < 0).
        # We need to integrate from 0 to x.
        # Let's create a grid that includes 0.
        
        # For simplicity and speed, let's assume we can evaluate the integral
        # by summing trapezoids from 0 to x for each point.
        # This is O(N^2) if done naively.
        # Better: Compute cumulative sum of trapezoids from x_min to x_max.
        # Then subtract.
        
        # Let's just use a simple loop or vectorized trapz if N is small.
        # N=100 is fine.
        
        D_app = np.zeros_like(a_vals)
        for i in range(len(a_vals)):
            if a_vals[i] == 1.0:
                D_app[i] = 1.0
            else:
                # Integrate from ln(a_vals[i]) to 0
                # Use a fine grid for integration? Or just the points?
                # The prompt implies the evaluator does this.
                # To match evaluator, we should probably use the same grid or a standard one.
                # Let's use the a_vals grid itself for integration, assuming it's dense enough.
                # If a_vals[i] < 1, integrate from x_i to 0.
                # Points in [x_i, 0] are a_vals[j] where log(a_vals[j]) >= x_i.
                
                mask = a_vals >= a_vals[i]
                if not np.any(mask):
                    D_app[i] = 1.0 # Should not happen
                    continue
                x_int = np.log(a_vals[mask])
                f_int = f_app[mask]
                
                # Add point 0 if not present
                if x_int[-1] < 0.0:
                    x_int = np.append(x_int, 0.0)
                    f_int = np.append(f_int, f_app[-1]) # Approximate f(1)
                
                # Integrate
                integral = -np.trapz(f_int, x_int)
                D_app[i] = np.exp(integral)
        
        return f_app, D_app, True

    def get_omega_m(Om0, w0, wa, a_vals):
        OmL0 = 1.0 - Om0
        term_L = (a_vals ** (-3.0 * (1.0 + w0 + wa))) * np.exp(3.0 * wa * a_vals)
        term_m = a_vals ** (-3.0)
        denom = Om0 * term_m + OmL0 * term_L
        return (Om0 * term_m) / denom

    # 3. Setup Optimization
    # Training set
    Om0_vals = np.array([0.25, 0.30])
    w0_vals = np.array([-1.2, -1.0, -0.8])
    wa_vals = np.array([-0.3, 0.0, 0.3])
    
    a_grid = np.linspace(0.01, 1.0, 50)
    
    # Pre-compute reference data
    ref_data = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                f_ref, D_ref = get_reference_growth(Om0, w0, wa, a_grid)
                if f_ref is not None:
                    ref_data.append((Om0, w0, wa, f_ref, D_ref))
    
    # Objective function
    def objective(c):
        max_err = 0.0
        for Om0, w0, wa, f_ref, D_ref in ref_data:
            f_app, D_app, valid = get_approx(Om0, w0, wa, c, a_grid)
            if not valid:
                return 1e6 # Penalty for gamma out of bounds
            
            # Relative errors
            err_f = np.abs(f_app - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_app - D_ref) / np.abs(D_ref)
            
            local_max = np.max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max
                
        return max_err

    # Initial guess
    c0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds
    bounds = [(-10, 10) for _ in range(6)]
    
    # Optimize
    best_c = c0
    best_score = objective(best_c)
    
    print(f"Initial score: {best_score}")
    
    if time.time() - start_time > time_limit * 0.2:
        return best_c
        
    try:
        res = minimize(objective, c0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success and res.fun < best_score:
            best_c = res.x
            best_score = res.fun
            print(f"Optimized score: {best_score}")
            
            # Polish
            if time.time() - start_time < time_limit * 0.8:
                 res2 = minimize(objective, best_c, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                 if res2.fun < best_score:
                     best_c = res2.x
                     best_score = res2.fun
                     print(f"Polished score: {best_score}")
                     
    except Exception as e:
        print(f"Optimization error: {e}")
        
    return best_c

# EVOLVE-BLOCK-END
