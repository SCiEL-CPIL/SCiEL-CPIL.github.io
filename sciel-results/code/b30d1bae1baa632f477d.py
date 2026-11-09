"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 90s total, aim to finish in 80s
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Define the representative validation cosmologies
    # The evaluator uses "disjoint public validation cosmologies".
    # We assume a spread of Omega_m, w0, wa.
    # Standard LambdaCDM is Om=0.3, w0=-1, wa=0.
    # We sample a grid to cover the likely space.
    
    om_m_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_values = [-1.2, -1.0, -0.8]
    wa_values = [-0.3, 0.0, 0.3]
    
    # Generate test models
    test_models = []
    for om in om_m_values:
        for w0 in w0_values:
            for wa in wa_values:
                test_models.append((om, w0, wa))
                
    # 2. Define the error calculation function
    # This mimics the evaluator's logic as described in the prompt.
    
    def calculate_error(coeffs, models):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_err_f = 0.0
        max_err_D = 0.0
        
        # We check a=1 and a range of a values for D.
        # The prompt says: "Maximize relative error of both f and normalized D"
        # f(a) = Om(a)^gamma(a)
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        
        # For each model, we calculate the reference f_ref and D_ref using the ODE.
        # And our approximation f_approx and D_approx.
        
        for om_m, w0, wa in models:
            # Reference calculation via ODE
            # f(a) = d ln D / d ln a
            # d f / d ln a + (3/2 (1+Om) + 2/3 w0 + 2/3 wa (1-a)) f - (1/2) Om = 0 ? 
            # Actually, the standard growth equation is:
            # D'' + (3/a + H') D' - (4 pi G / 3) rho D = 0
            # In terms of f = d ln D / d ln a:
            # f' + f(2 + f) + (3/2)(1 + Om(a)) = 0  -- Wait, this is for matter domination?
            # General CPL:
            # f' + f^2 + f * (3/2 * (1 + Om(a)) + 2/3 * w0 + 2/3 * wa * (1-a)) - 1/2 * Om(a) = 0
            # Let's use this ODE for the reference.
            
            def ode_rhs(ln_a, f):
                a = np.exp(ln_a)
                # Om(a) = Om_m * a^3 / (Om_m * a^3 + (1-Om_m) * exp(3*(1+w0+wa)*(-ln(a)) ) )
                # Note: exp(3*(1+w0+wa)*ln(a)) = a^(3*(1+w0+wa))
                # Let E(a)^2 = Om_m a^3 + (1-Om_m) a^(3(1+w0+wa))
                # Om(a) = Om_m a^3 / E(a)^2
                
                exp_wa = 3 * (1 + w0 + wa)
                de_term = (1 - om_m) * (a ** exp_wa)
                mat_term = om_m * (a ** 3)
                e2 = mat_term + de_term
                om_a = mat_term / e2 if e2 > 0 else 0.0
                
                # ODE term: f' = -f^2 - f * (3/2 (1+Om) + 2/3 w0 + 2/3 wa (1-a)) + 1/2 Om
                coeff = 1.5 * (1 + om_a) + (2.0/3.0) * w0 + (2.0/3.0) * wa * (1 - a)
                return -f**2 - f * coeff + 0.5 * om_a

            # Integrate from a=1 (ln_a=0) downwards to a_min
            # We need D(a)/D(1) = exp(-int_{ln a}^0 f(x) dx)
            # So we need f(x) for x in [ln(a), 0].
            # We can integrate the ODE backwards from ln_a=0 to ln_a=a_min.
            
            a_min = 0.05
            ln_a_min = np.log(a_min)
            
            # Solve ODE on grid
            ln_a_grid = np.linspace(0, ln_a_min, 200)
            f_grid = np.zeros_like(ln_a_grid)
            f_grid[0] = 0.55 # Initial guess at a=1, or solve for it?
            # At a=1, f(1) is determined by the model. 
            # For LambdaCDM, f(1) ~ 0.55.
            # Let's solve the ODE for f(1) by integrating from a=0.1 to a=1?
            # Or just assume f(1) is the value that satisfies the algebraic part?
            # Actually, the ODE is first order. We need an initial condition.
            # The standard approximation f(1) = Om(1)^0.55 is often used, but for reference we need high accuracy.
            # Let's integrate from a=0.1 where f is well-behaved?
            # Better: Use the fact that at a=1, we can estimate f(1) from the ODE if we assume it's close to steady state? No.
            # Let's integrate from a=0.1 to a=1.
            
            # Re-integrate from a=0.1 to a=1 to get f(1) accurately?
            # No, we need f(a) for all a.
            # Let's integrate from a=0.1 to a=1, then from a=1 to a=0.1?
            # Actually, we can just integrate from a=0.1 to a=1 and store f.
            # Then for D(a), we integrate f from a to 1.
            
            # Let's set up the integration for the reference f and D.
            # We will compute f_ref(a) and D_ref(a) on a grid of a values.
            
            a_eval = np.linspace(a_min, 1.0, 50)
            ln_a_eval = np.log(a_eval)
            
            # Solve ODE from a=0.1 to a=1.0
            # We need an initial condition at a=0.1.
            # At small a, matter dominates, f ~ 1.
            # Let's start at a=0.1 with f=0.95 (approx).
            # Actually, let's use a solver.
            
            from scipy.integrate import solve_ivp
            
            def sol_func(t, y):
                return [ode_rhs(t, y[0])]
                
            # Integrate from ln(0.1) to 0
            t_span = (np.log(0.1), 0)
            sol = solve_ivp(sol_func, t_span, [0.95], t_eval=np.sort(ln_a_eval)[::-1], rtol=1e-8, atol=1e-10)
            
            if not sol.success:
                return 1e10 # Penalty for failure
                
            # sol.y[0] is f at sol.t
            # We need f at a_eval. sol.t is sorted ln_a.
            # Let's interpolate f_ref to a_eval
            
            # Note: solve_ivp with t_eval returns values at t_eval.
            # We passed t_eval=sorted(ln_a_eval)[::-1] which is from 0 to ln(0.1)?
            # No, t_span is (ln0.1, 0). t_eval must be within t_span.
            # Let's just use t_eval=np.linspace(np.log(0.1), 0, 50)
            
            t_eval_ref = np.linspace(np.log(0.1), 0, 100)
            sol_ref = solve_ivp(sol_func, (np.log(0.1), 0), [0.95], t_eval=t_eval_ref, rtol=1e-8, atol=1e-10)
            
            if not sol_ref.success:
                return 1e10
                
            f_ref_grid = sol_ref.y[0]
            ln_a_ref_grid = sol_ref.t
            
            # Interpolate f_ref to a_eval
            f_ref = np.interp(ln_a_eval, ln_a_ref_grid, f_ref_grid)
            
            # Calculate D_ref
            # D(a)/D(1) = exp(-int_{ln a}^0 f(x) dx)
            # We have f on ln_a grid.
            # int_{ln a}^0 f(x) dx = - int_0^{ln a} f(x) dx
            # So D(a)/D(1) = exp(int_0^{ln a} f(x) dx)
            
            # Cumulative integral of f_ref from 0 to ln_a
            # Since ln_a_eval is negative (a<1), we integrate from 0 down to ln_a.
            # Let's compute cumulative integral from the end (0) backwards.
            
            # Sort ln_a_ref_grid descending? No, it's ascending from ln(0.1) to 0.
            # We want integral from 0 to x.
            # Let's flip the arrays so they go from 0 to ln(0.1)
            ln_a_rev = ln_a_ref_grid[::-1]
            f_ref_rev = f_ref_grid[::-1]
            
            # Integrate from 0
            # quad on each point is slow. Use cumulative trapezoid.
            # np.cumtrapz(f, x, initial=0)
            # But we need integral from 0 to x.
            # If we reverse, x goes 0 -> -inf.
            # Let's just use quad for the 50 points. It's fast enough.
            
            D_ref = np.ones_like(a_eval)
            for i, la in enumerate(ln_a_eval):
                # Integrate f_ref from 0 to la
                # f_ref is defined on ln_a_ref_grid (ascending)
                # We need to handle the interpolation for the integrand
                def integrand(x):
                    return np.interp(x, ln_a_ref_grid, f_ref_grid)
                    
                val, _ = quad(integrand, 0, la, limit=50)
                D_ref[i] = np.exp(val)
            
            # Now calculate Approximation
            # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
            # q = 1 - Om(a)
            # u = 1 + w0 + wa*(1-a)
            
            max_err_local = 0.0
            
            for i, a in enumerate(a_eval):
                # Calculate Om(a) for this model
                exp_wa = 3 * (1 + w0 + wa)
                de_term = (1 - om_m) * (a ** exp_wa)
                mat_term = om_m * (a ** 3)
                e2 = mat_term + de_term
                om_a = mat_term / e2 if e2 > 0 else 0.0
                
                q = 1 - om_a
                u = 1 + w0 + wa * (1 - a)
                
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10 # Invalid gamma
                    
                f_approx = om_a ** gamma
                
                # Relative error for f
                if f_ref[i] != 0:
                    err_f = abs(f_approx - f_ref[i]) / abs(f_ref[i])
                else:
                    err_f = 0.0
                    
                # Relative error for D
                # D_approx = exp(-int_{ln a}^0 f_approx(x) dx)
                # We need to integrate f_approx(x) from 0 to ln_a
                # f_approx(x) depends on a=exp(x)
                
                def integrand_approx(x):
                    ax = np.exp(x)
                    # Recalculate om_ax
                    exp_wa_x = 3 * (1 + w0 + wa)
                    de_term_x = (1 - om_m) * (ax ** exp_wa_x)
                    mat_term_x = om_m * (ax ** 3)
                    e2_x = mat_term_x + de_term_x
                    om_ax = mat_term_x / e2_x if e2_x > 0 else 0.0
                    
                    q_x = 1 - om_ax
                    u_x = 1 + w0 + wa * (1 - ax)
                    gamma_x = c0 + c1*q_x + c2*q_x**2 + c3*u_x + c4*wa*ax + c5*u_x*q_x
                    
                    # Check bounds? If we fail here, we return large error.
                    if gamma_x < 0.1 or gamma_x > 1.2:
                        return 1e10 # This will cause quad to fail or return large number?
                        # Better to check bounds outside.
                        
                    return om_ax ** gamma_x
                
                # Check bounds for the whole range [la, 0]
                # We can do a quick check on the grid
                a_check = np.linspace(a, 1.0, 10)
                for ac in a_check:
                    exp_wa_c = 3 * (1 + w0 + wa)
                    de_c = (1 - om_m) * (ac ** exp_wa_c)
                    mat_c = om_m * (ac ** 3)
                    e2_c = de_c + mat_c
                    om_c = mat_c / e2_c if e2_c > 0 else 0.0
                    q_c = 1 - om_c
                    u_c = 1 + w0 + wa * (1 - ac)
                    gamma_c = c0 + c1*q_c + c2*q_c**2 + c3*u_c + c4*wa*ac + c5*u_c*q_c
                    if gamma_c < 0.1 or gamma_c > 1.2:
                        return 1e10
                        
                val_approx, _ = quad(integrand_approx, 0, la, limit=50)
                D_approx = np.exp(val_approx)
                
                if D_ref[i] != 0:
                    err_D = abs(D_approx - D_ref[i]) / abs(D_ref[i])
                else:
                    err_D = 0.0
                    
                err_local = max(err_f, err_D)
                if err_local > max_err_local:
                    max_err_local = err_local
                    
            if max_err_local > max_err_f:
                max_err_f = max_err_local
            if max_err_local > max_err_D:
                max_err_D = max_err_local
                
        # The score is 100 * max(relative error)
        # We want to minimize this.
        return 100.0 * max(max_err_f, max_err_D)

    # 3. Optimization
    # Initial guess: constant gamma 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds: [-10, 10]
    bounds = [(-10, 10)] * 6
    
    best_x = x0
    best_score = calculate_error(x0, test_models)
    
    # Use Nelder-Mead
    # We need to be careful with the time.
    # calculate_error is expensive (ODE + Quad).
    # Let's limit the number of iterations.
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        return calculate_error(x, test_models)
        
    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.fun < best_score:
        best_score = res.fun
        best_x = res.x
        
    # One more refinement step if time allows?
    # Nelder-Mead usually converges well.
    
    # Ensure coefficients are finite and within bounds
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

# EVOLVE-BLOCK-END
