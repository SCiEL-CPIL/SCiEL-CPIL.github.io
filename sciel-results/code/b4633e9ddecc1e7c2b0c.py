"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
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
    # Use a finer grid for better accuracy
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
            sol = solve_ivp(ode_rhs, t_span, [f_init], t_eval=t_eval, rtol=1e-9, atol=1e-11)
            if not sol.success:
                continue
                
            f_ref = sol.y[0]
            
            # Calculate D_ref using cumulative_trapezoid
            # We integrate from 0 to ln_a. 
            # ln_a_eval is ascending (negative to 0).
            # We want integral from 0 to x (where x is negative).
            # Let's reverse to go from 0 down to ln(0.1).
            ln_a_rev = ln_a_eval[::-1]
            f_ref_rev = f_ref[::-1]
            
            # cumulative_trapezoid(y, x, initial=0)
            # This computes integral from x[0] to x[i].
            # x[0] = 0, x[-1] = ln(0.1).
            int_vals = cumulative_trapezoid(f_ref_rev, ln_a_rev, initial=0.0)
            
            # Reverse back to match ln_a_eval order
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

    # Pre-compute common terms for speed
    om_a_grid = []
    q_grid = []
    u_grid = []
    wa_a_grid = []
    uq_grid = []
    
    for data in ref_data:
        om_m = data['om_m']
        w0 = data['w0']
        wa = data['wa']
        
        exp_wa = 3 * (1 + w0 + wa)
        de_term = (1 - om_m) * (a_eval ** exp_wa)
        mat_term = om_m * (a_eval ** 3)
        e2 = mat_term + de_term
        om_a = mat_term / e2
        
        q = 1 - om_a
        u = 1 + w0 + wa * (1 - a_eval)
        wa_a = wa * a_eval
        uq = u * q
        
        om_a_grid.append(om_a)
        q_grid.append(q)
        u_grid.append(u)
        wa_a_grid.append(wa_a)
        uq_grid.append(uq)
        
        data['om_a'] = om_a
        data['q'] = q
        data['u'] = u
        data['wa_a'] = wa_a
        data['uq'] = uq

    def calculate_error_components(coeffs):
        """Returns list of individual error values for smooth optimization."""
        c0, c1, c2, c3, c4, c5 = coeffs
        all_errors = []
        
        for i, data in enumerate(ref_data):
            om_a = data['om_a']
            q = data['q']
            u = data['u']
            wa_a = data['wa_a']
            uq = data['uq']
            f_ref = data['f_ref']
            D_ref = data['D_ref']
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa_a + c5*uq
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return None
                
            f_approx = om_a ** gamma
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            all_errors.extend(err_f)
            
            # Calculate D_approx
            ln_a_rev = ln_a_eval[::-1]
            f_approx_rev = f_approx[::-1]
            
            int_vals = cumulative_trapezoid(f_approx_rev, ln_a_rev, initial=0.0)
            int_vals_rev = int_vals[::-1]
            
            D_approx = np.exp(int_vals_rev)
            
            err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
            all_errors.extend(err_D)
        
        return np.array(all_errors)

    def calculate_error_fast(coeffs):
        """Calculates the max relative error (scaled by 100)."""
        errors = calculate_error_components(coeffs)
        if errors is None:
            return 1e10
        return 100.0 * np.max(errors)

    def smooth_objective(x, beta=100.0):
        """Log-sum-exp smooth approximation of max error."""
        if time.time() > deadline:
            return 1e10
        
        errors = calculate_error_components(x)
        if errors is None:
            return 1e10
        
        # Log-sum-exp for smooth max
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        
        # Smooth max: (1/beta) * log(sum(exp(beta * err)))
        shifted = errors - max_err
        lse = max_err + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
        
        return 100.0 * lse

    # 4. Optimization
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-10, 10)] * 6
    
    best_x = x0
    best_score = calculate_error_fast(x0)
    
    # Multi-start optimization
    # Generate several starting points
    np.random.seed(42)
    starts = [x0]
    for _ in range(4):
        # Small perturbations
        perturbation = np.random.normal(0, 0.1, 6)
        starts.append(np.clip(np.array(x0) + perturbation, -10, 10))
        
    for start in starts:
        if time.time() > deadline - 20.0:
            break
            
        # Stage 1: L-BFGS-B with smooth objective
        def objective_smooth(x):
            if time.time() > deadline:
                return 1e10
            return smooth_objective(x, beta=100.0)
        
        try:
            res1 = minimize(objective_smooth, start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res1.fun < best_score:
                # Verify with exact objective
                exact_score = calculate_error_fast(res1.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_x = res1.x
        except Exception:
            pass
        
        # Stage 2: Nelder-Mead polish on exact objective
        if time.time() > deadline - 5.0:
            continue
            
        def objective_exact(x):
            if time.time() > deadline:
                return 1e10
            return calculate_error_fast(x)
            
        try:
            res2 = minimize(objective_exact, best_x, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Final check and clip
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

# EVOLVE-BLOCK-END
