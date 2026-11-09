"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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
    # Use a denser grid for better accuracy
    a_eval = np.linspace(0.1, 1.0, 200)
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
        a_init = 0.1
        ln_a_init = np.log(a_init)
        exp_wa_init = 3 * (1 + w0 + wa)
        de_term_init = (1 - om_m) * (a_init ** exp_wa_init)
        mat_term_init = om_m * (a_init ** 3)
        e2_init = mat_term_init + de_term_init
        om_a_init = mat_term_init / e2_init if e2_init > 0 else 0.0
        f_init = om_a_init ** 0.55
        
        t_span = (ln_a_init, 0.0)
        t_eval = ln_a_eval
        
        try:
            sol = solve_ivp(ode_rhs, t_span, [f_init], t_eval=t_eval, rtol=1e-8, atol=1e-10)
            if not sol.success:
                continue
                
            f_ref = sol.y[0]
            
            # Calculate D_ref using cumulative integration
            ln_a_rev = ln_a_eval[::-1]
            f_ref_rev = f_ref[::-1]
            
            int_vals = np.concatenate(([0.0], np.cumtrapz(f_ref_rev, ln_a_rev)))
            int_vals_rev = int_vals[::-1]
            
            D_ref = np.exp(int_vals_rev)
            
            # Also store log versions for Chebyshev fitting
            ln_f_ref = np.log(f_ref)
            ln_D_ref = np.log(D_ref)
            
            # Pre-compute Om(a), q, u for this model
            exp_wa = 3 * (1 + w0 + wa)
            de_term = (1 - om_m) * (a_eval ** exp_wa)
            mat_term = om_m * (a_eval ** 3)
            e2 = mat_term + de_term
            om_a = mat_term / e2
            
            q = 1 - om_a
            u = 1 + w0 + wa * (1 - a_eval)
            
            ref_data.append({
                'om_m': om_m,
                'w0': w0,
                'wa': wa,
                'f_ref': f_ref,
                'D_ref': D_ref,
                'ln_f_ref': ln_f_ref,
                'ln_D_ref': ln_D_ref,
                'om_a': om_a,
                'q': q,
                'u': u
            })
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Pre-compute common arrays for speed
    wa_a_eval = np.array([wa * a_eval for data in ref_data for wa in [data['wa']]])
    # Actually, let's just compute on the fly but with pre-computed om_a, q, u

    # 3. Define fast error calculation using pre-computed reference data
    def calculate_error_fast(coeffs):
        """Exact minimax error: 100 * max relative error over f and D."""
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for data in ref_data:
            om_a = data['om_a']
            q = data['q']
            u = data['u']
            wa = data['wa']
            f_ref = data['f_ref']
            D_ref = data['D_ref']
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_eval + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
                
            f_approx = om_a ** gamma
            
            # Error in f (relative)
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            max_err = max(max_err, np.max(err_f))
            
            # Calculate D_approx
            ln_a_rev = ln_a_eval[::-1]
            f_approx_rev = f_approx[::-1]
            
            int_vals = np.concatenate(([0.0], np.cumtrapz(f_approx_rev, ln_a_rev)))
            int_vals_rev = int_vals[::-1]
            
            D_approx = np.exp(int_vals_rev)
            
            # Error in D (relative)
            err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
            max_err = max(max_err, np.max(err_D))
        
        return 100.0 * max_err

    def calculate_log_errors(coeffs):
        """Returns log-space errors for Chebyshev fitting.
        Returns (max_log_err_f, max_log_err_D, all_errors) or None if gamma out of bounds."""
        c0, c1, c2, c3, c4, c5 = coeffs
        all_errors = []
        
        for data in ref_data:
            om_a = data['om_a']
            q = data['q']
            u = data['u']
            wa = data['wa']
            ln_f_ref = data['ln_f_ref']
            ln_D_ref = data['ln_D_ref']
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_eval + c5*u*q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return None
                
            f_approx = om_a ** gamma
            ln_f_approx = np.log(f_approx)
            
            # Log error in f: ln(f_approx/f_ref) = ln_f_approx - ln_f_ref
            log_err_f = np.abs(ln_f_approx - ln_f_ref)
            all_errors.extend(log_err_f)
            
            # Calculate D_approx
            ln_a_rev = ln_a_eval[::-1]
            f_approx_rev = f_approx[::-1]
            
            int_vals = np.concatenate(([0.0], np.cumtrapz(f_approx_rev, ln_a_rev)))
            int_vals_rev = int_vals[::-1]
            
            ln_D_approx = int_vals_rev
            log_err_D = np.abs(ln_D_approx - ln_D_ref)
            all_errors.extend(log_err_D)
        
        return np.array(all_errors)

    def chebyshev_objective(x, beta=100.0):
        """Smooth Chebyshev objective using log-sum-exp on log-space errors."""
        if time.time() > deadline:
            return 1e10
        
        errors = calculate_log_errors(x)
        if errors is None:
            return 1e10
        
        if len(errors) == 0:
            return 1e10
            
        # Log-sum-exp smooth max
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        
        # Smooth max: (1/beta) * log(sum(exp(beta * err)))
        shifted = errors - max_err
        lse = max_err + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
        
        return 100.0 * lse

    def exact_minimax_objective(x):
        """Exact minimax objective for final polish."""
        if time.time() > deadline:
            return 1e10
        return calculate_error_fast(x)

    # 4. Multi-stage optimization with Chebyshev formulation
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-10, 10)] * 6
    
    best_x = x0
    best_score = calculate_error_fast(x0)
    
    # Stage 1: Coarse search with smooth Chebyshev objective
    if time.time() < deadline - 30.0:
        def objective_coarse(x):
            if time.time() > deadline:
                return 1e10
            return chebyshev_objective(x, beta=50.0)
        
        try:
            res1 = minimize(objective_coarse, x0, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            if res1.fun < best_score:
                exact_score = calculate_error_fast(res1.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_x = res1.x
        except Exception:
            pass
    
    # Stage 2: Medium-precision Chebyshev with higher beta
    if time.time() < deadline - 20.0:
        def objective_medium(x):
            if time.time() > deadline:
                return 1e10
            return chebyshev_objective(x, beta=100.0)
        
        try:
            res2 = minimize(objective_medium, best_x, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res2.fun < best_score:
                exact_score = calculate_error_fast(res2.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_x = res2.x
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B on smooth Chebyshev for gradient-based refinement
    if time.time() < deadline - 15.0:
        def objective_lbfgs(x):
            if time.time() > deadline:
                return 1e10
            return chebyshev_objective(x, beta=200.0)
        
        try:
            res3 = minimize(objective_lbfgs, best_x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6})
            if res3.fun < best_score:
                exact_score = calculate_error_fast(res3.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_x = res3.x
        except Exception:
            pass
    
    # Stage 4: Fine Nelder-Mead with exact minimax
    if time.time() < deadline - 10.0:
        try:
            res4 = minimize(exact_minimax_objective, best_x, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res4.fun < best_score:
                best_score = res4.fun
                best_x = res4.x
        except Exception:
            pass
    
    # Stage 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            res5 = minimize(exact_minimax_objective, best_x, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res5.fun < best_score:
                best_score = res5.fun
                best_x = res5.x
        except Exception:
            pass

    # Final check and clip
    best_x = np.clip(best_x, -10, 10)
    
    # Verify the final result
    final_score = calculate_error_fast(best_x)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_x)

# EVOLVE-BLOCK-END
