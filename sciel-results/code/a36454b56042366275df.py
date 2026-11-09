"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 90s total, aim to finish in 85s
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Define the representative validation cosmologies
    om_m_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_values = [-1.2, -1.0, -0.8]
    wa_values = [-0.3, 0.0, 0.3]
    
    test_models = []
    for om in om_m_values:
        for w0 in w0_values:
            for wa in wa_values:
                test_models.append((om, w0, wa))
                
    # 2. Pre-compute reference data for all models
    # We will store: a_eval (common grid), and for each model: f_ref, D_ref
    a_eval = np.linspace(0.1, 1.0, 100)
    ln_a_eval = np.log(a_eval)
    
    # Store reference data in lists
    ref_data = []
    
    for om_m, w0, wa in test_models:
        # Define ODE RHS
        def ode_rhs(ln_a, f):
            a = np.exp(ln_a)
            exp_wa = 3 * (1 + w0 + wa)
            de_term = (1 - om_m) * (a ** exp_wa)
            mat_term = om_m * (a ** 3)
            e2 = mat_term + de_term
            om_a = mat_term / e2 if e2 > 0 else 0.0
            
            coeff = 1.5 * (1 + om_a) + (2.0/3.0) * w0 + (2.0/3.0) * wa * (1 - a)
            return -f**2 - f * coeff + 0.5 * om_a

        # Solve ODE from a=0.1 to a=1.0
        # Initial condition at a=0.1: f ~ 1.0 (matter domination)
        # Using a slightly better initial guess: f(0.1) ~ Om(0.1)^0.55
        a_init = 0.1
        ln_a_init = np.log(a_init)
        exp_wa_init = 3 * (1 + w0 + wa)
        de_term_init = (1 - om_m) * (a_init ** exp_wa_init)
        mat_term_init = om_m * (a_init ** 3)
        e2_init = mat_term_init + de_term_init
        om_a_init = mat_term_init / e2_init if e2_init > 0 else 0.0
        f_init = om_a_init ** 0.55
        
        t_span = (ln_a_init, 0.0)
        t_eval = ln_a_eval # This is sorted ascending from ln(0.1) to 0
        
        try:
            sol = solve_ivp(ode_rhs, t_span, [f_init], t_eval=t_eval, rtol=1e-8, atol=1e-10)
            if not sol.success:
                continue
                
            f_ref = sol.y[0]
            
            # Calculate D_ref
            # D(a)/D(1) = exp(-int_{ln a}^0 f(x) dx) = exp(int_0^{ln a} f(x) dx)
            # We have f_ref on ln_a_eval (ascending).
            # We need integral from 0 to ln_a.
            # Let's reverse the arrays to integrate from 0 downwards.
            ln_a_rev = ln_a_eval[::-1]
            f_ref_rev = f_ref[::-1]
            
            # Cumulative trapezoidal integration
            # integral from 0 to x. Since x is negative, we integrate backwards.
            # np.cumtrapz(f, x) gives integral from x[0] to x[i].
            # Here x[0]=0, x[-1]=ln(0.1).
            # So cumtrapz gives int_0^{x[i]} f dx.
            int_vals = np.concatenate(([0.0], np.trapezoid(f_ref_rev, ln_a_rev, axis=0)))
            
            # int_vals corresponds to ln_a_rev. We need to reverse it back to match ln_a_eval order.
            int_vals_rev = int_vals[::-1]
            
            D_ref = np.exp(int_vals_rev)
            
            ref_data.append({
                'om_m': om_m,
                'w0': w0,
                'wa': wa,
                'f_ref': f_ref,
                'D_ref': D_ref
            })
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 3. Define fast error calculation using pre-computed reference data
    def calculate_error_fast(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for data in ref_data:
            om_m = data['om_m']
            w0 = data['w0']
            wa = data['wa']
            f_ref = data['f_ref']
            D_ref = data['D_ref']
            
            # Calculate Om(a) for the grid
            exp_wa = 3 * (1 + w0 + wa)
            de_term = (1 - om_m) * (a_eval ** exp_wa)
            mat_term = om_m * (a_eval ** 3)
            e2 = mat_term + de_term
            om_a = mat_term / e2
            
            q = 1 - om_a
            u = 1 + w0 + wa * (1 - a_eval)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_eval + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
                
            f_approx = om_a ** gamma
            
            # Error in f
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            max_err = max(max_err, np.max(err_f))
            
            # Calculate D_approx
            # D_approx(a) = exp(-int_{ln a}^0 f_approx(x) dx)
            # We need to integrate f_approx(x) where x is ln(a).
            # f_approx(x) = Om(exp(x))^gamma(exp(x))
            # We can approximate the integral using the trapezoidal rule on the grid.
            
            # We need f_approx as a function of ln_a.
            # We already have f_approx on a_eval, which corresponds to ln_a_eval.
            # We want int_0^{ln_a} f_approx(x) dx.
            # Similar to reference calculation:
            ln_a_rev = ln_a_eval[::-1]
            f_approx_rev = f_approx[::-1]
            
            int_vals = np.concatenate(([0.0], np.trapezoid(f_approx_rev, ln_a_rev, axis=0)))
            int_vals_rev = int_vals[::-1]
            
            D_approx = np.exp(int_vals_rev)
            
            # Error in D
            err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
            max_err = max(max_err, np.max(err_D))
            
        return 100.0 * max_err

    # 4. Optimization
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-10, 10)] * 6
    
    best_x = x0
    best_score = calculate_error_fast(x0)
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        return calculate_error_fast(x)
        
    # Use Nelder-Mead with limited iterations
    # The error function is now fast (vectorized), so we can afford more iterations
    try:
        res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass

    # Final check and clip
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

# EVOLVE-BLOCK-END
