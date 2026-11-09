import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Time budget guard
    start_time = time.time()
    time_limit = 80.0  # Leave 10s margin

    # 1. Define validation grid
    # Cosmologies: Om0 in [0.2, 0.35], w0 in [-1.2, 0.0], wa in [-0.5, 0.5]
    om0_vals = np.linspace(0.2, 0.35, 5)
    w0_vals = np.linspace(-1.2, 0.0, 5)
    wa_vals = np.linspace(-0.5, 0.5, 5)
    
    a_vals = np.linspace(0.1, 1.0, 20)
    
    # Precompute references
    # We will store f_ref and D_ref for each combo
    # To save memory and time, we process one combo at a time in the objective?
    # No, precomputing is faster if the grid is small. 5*5*5 = 125 combos. 20 a-points. 2500 points.
    # This is small. We can precompute all refs.
    
    refs = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Calculate H^2 terms
                # H^2 = H0^2 (Om_m a^-3 + Om_L a^-3(1+w0+wa(1-a)))
                # We only need ratios, so H0=1.
                
                # Function to get Om_m(a) and H terms
                def get_cosmo_params(a):
                    u = 1.0 + w0 + wa * (1.0 - a)
                    om_m = om0 * a**(-3)
                    om_l = (1.0 - om0) * a**(-3 * u)
                    h2 = om_m + om_l
                    om_m_a = om_m / h2
                    
                    # a * H' / H
                    # 2 H H' = d/da H^2 = d/da (om_m + om_l)
                    # d/da om_m = -3 om_m / a
                    # d/da om_l = -3 u om_l / a + om_l * (-3) * da/da (1+w0+wa(1-a))
                    # da/da (1+w0+wa(1-a)) = -wa
                    # d/da om_l = -3 u om_l / a + om_l * (-3) * (-wa) = -3 om_l / a * (u - wa)
                    
                    dh2_da = -3.0 * om_m / a - 3.0 * om_l / a * (u - wa)
                    a_hp_h = 0.5 * a * dh2_da / h2
                    
                    return om_m_a, a_hp_h, u

                # Integrate f ODE backwards from a=1
                # f' = 1/a * [ 1.5 Om_m - f^2 - 0.5 f - a_hp_h * f ]
                # Initial f(1) ~ Om_m(1)^0.55
                om_m_1, _, _ = get_cosmo_params(1.0)
                f_init = om_m_1**0.55
                
                # Use simple RK4 or Euler for speed? 
                # a_vals is 0.1 to 1.0. Let's integrate from 1.0 down to 0.1.
                a_int = np.linspace(1.0, 0.1, 100)
                f_int = np.zeros_like(a_int)
                f_int[0] = f_init
                
                # Step backwards
                for i in range(len(a_int) - 1):
                    a_curr = a_int[i]
                    a_next = a_int[i+1]
                    da = a_next - a_curr # negative
                    
                    om_m_a, a_hp_h, u = get_cosmo_params(a_curr)
                    f_curr = f_int[i]
                    
                    def df_da(a, f):
                        om_m_a, a_hp_h, u = get_cosmo_params(a)
                        return (1.5 * om_m_a - f**2 - 0.5 * f - a_hp_h * f) / a
                    
                    # RK4
                    k1 = df_da(a_curr, f_curr)
                    k2 = df_da(a_curr + 0.5*da, f_curr + 0.5*da*k1)
                    k3 = df_da(a_curr + 0.5*da, f_curr + 0.5*da*k2)
                    k4 = df_da(a_curr + da, f_curr + da*k3)
                    f_int[i+1] = f_curr + da/6.0 * (k1 + 2*k2 + 2*k3 + k4)
                    
                # Interpolate f_ref to a_vals
                f_ref = np.interp(a_vals, a_int[::-1], f_int[::-1])
                
                # Calculate D_ref
                # ln D(a) - ln D(1) = - integral_a^1 f(a') da'/a'
                # Let x = ln a. dx = da/a.
                # integral from a to 1 of f(a)/a da = integral from ln a to 0 f(e^x) dx
                # We have f_ref at a_vals. We can integrate numerically.
                # D(a)/D(1) = exp( - integral_a^1 f(a')/a' da' )
                # Using trapezoidal rule on a_vals? Better to integrate on the finer grid a_int.
                # f_int is defined on a_int (1.0 to 0.1).
                # integral from a to 1 of f(a)/a da
                # Let's compute cumulative integral from 1.0 down to a.
                # I(a) = integral_a^1 f(a')/a' da'
                # Using a_int:
                inv_a = 1.0 / a_int
                integrand = f_int * inv_a
                # Cumulative sum from end (a=0.1) to start (a=1.0)?
                # We want integral from a to 1.
                # Let's reverse arrays to go 0.1 -> 1.0
                a_rev = a_int[::-1]
                f_rev = f_int[::-1]
                integrand_rev = f_rev * (1.0 / a_rev)
                
                # Cumulative integral from 0.1 to a
                # scipy.integrate.cumulative_trapezoid
                from scipy.integrate import cumulative_trapezoid
                cum_int = cumulative_trapezoid(integrand_rev, a_rev, initial=0.0)
                # cum_int[i] is integral from 0.1 to a_rev[i]
                # We want integral from a to 1.0.
                # Total integral from 0.1 to 1.0 is cum_int[-1]
                # Integral from a to 1.0 = Total - Integral from 0.1 to a
                total_int = cum_int[-1]
                # Interpolate cum_int to a_vals
                cum_int_at_a = np.interp(a_vals, a_rev, cum_int)
                integral_a_to_1 = total_int - cum_int_at_a
                D_ref = np.exp(-integral_a_to_1)
                
                refs.append((f_ref, D_ref))

    # 2. Define Objective Function
    def objective(c):
        c0, c1, c2, c3, c4, c5 = c
        max_err = 0.0
        penalty = 0.0
        
        for idx, (f_ref, D_ref) in enumerate(refs):
            om0 = om0_vals[idx // 25]
            w0 = w0_vals[(idx // 5) % 5]
            wa = wa_vals[idx % 5]
            
            # Calculate model
            # We need Om_m(a) for each a in a_vals for this cosmology
            # We can reuse the logic from precomputation, but it's faster to precompute Om_m_a for the grid too.
            # Let's precompute Om_m_a for the grid to avoid recalculating.
            # Actually, we didn't store Om_m_a. Let's just recalculate or store it.
            # For simplicity and speed in the inner loop, let's precompute a dictionary of Om_m_a.
            # But we are in the objective function which is called many times.
            # It is better to precompute Om_m_a for all refs.
            
            # Let's refactor: Precompute Om_m_a, u, etc. for the grid.
            # We will do this in the setup phase.
            
            # For now, let's assume we have a helper to get Om_m_a.
            # Since we didn't store it, we will recalculate it here. It's cheap.
            
            for i, a in enumerate(a_vals):
                u = 1.0 + w0 + wa * (1.0 - a)
                om_m = om0 * a**(-3)
                om_l = (1.0 - om0) * a**(-3 * u)
                h2 = om_m + om_l
                om_m_a = om_m / h2
                q = 1.0 - om_m_a
                
                gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
                
                # Check bounds
                if gamma < 0.1 or gamma > 1.2:
                    penalty += 1e6 * (max(0.1 - gamma, 0) + max(gamma - 1.2, 0))
                
                f_model = om_m_a**gamma
                D_model = np.exp(-integral_a_to_1) # Wait, D_model depends on integral of f_model.
                # We need to compute the integral for D_model.
                # This is expensive inside the loop.
                # We must integrate f_model over a_vals.
                
                # This approach is too slow. Integrating D for every objective call is O(N_cosmo * N_a * N_a).
                # We need a faster way or a smaller grid.
                
                # Let's reduce grid size for optimization.
                # Or, approximate the integral.
                # D(a) = exp(-int_a^1 f(a')/a' da')
                # If we have f_model at a_vals, we can use cumulative trapezoid.
                # But doing this for 125 cosmologies * 20 points * many iterations is slow.
                
                # Let's reduce the grid.
                # 3x3x3 = 27 cosmologies. 10 a-points.
                # 270 points.
                # Objective call: 270 integrals? No, 27 integrals of length 10.
                # That's fast.
                
                pass # Placeholder for logic
        
        # Return max error + penalty
        # We need to re-implement the loop with the actual logic.
        
        # Let's restart the logic with a smaller grid and precomputed integrals structure.
        return 0.0

    # Re-do setup with smaller grid
    om0_vals = np.linspace(0.2, 0.35, 3)
    w0_vals = np.linspace(-1.2, 0.0, 3)
    wa_vals = np.linspace(-0.5, 0.5, 3)
    a_vals = np.linspace(0.1, 1.0, 10)
    
    refs = []
    om_m_a_grid = []
    u_grid = []
    a_hp_h_grid = []
    
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                f_ref, D_ref, om_m_a_arr, u_arr, a_hp_h_arr = calc_reference_fast(om0, w0, wa, a_vals)
                refs.append((f_ref, D_ref))
                om_m_a_grid.append(om_m_a_arr)
                u_grid.append(u_arr)
                a_hp_h_grid.append(a_hp_h_arr)
                
    def objective(c):
        c0, c1, c2, c3, c4, c5 = c
        max_err = 0.0
        penalty = 0.0
        
        for idx in range(len(refs)):
            f_ref, D_ref = refs[idx]
            om_m_a_arr = om_m_a_grid[idx]
            u_arr = u_grid[idx]
            
            # Calculate f_model
            q = 1.0 - om_m_a_arr
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u_arr + c4 * wa_vals[idx % 3] * a_vals + c5 * u_arr * q
            
            # Check bounds
            out_of_bounds = (gamma < 0.1) | (gamma > 1.2)
            if np.any(out_of_bounds):
                penalty += 1e5 * np.sum(np.abs(np.clip(gamma, 0.1, 1.2) - gamma))
            
            f_model = om_m_a_arr**gamma
            
            # Calculate D_model
            # integral from a to 1 of f(a)/a da
            # Use cumulative trapezoid on a_vals
            integrand = f_model / a_vals
            # Integrate from a to 1.
            # Reverse a_vals to go 0.1 -> 1.0
            a_rev = a_vals[::-1]
            integrand_rev = integrand[::-1]
            from scipy.integrate import cumulative_trapezoid
            cum_int = cumulative_trapezoid(integrand_rev, a_rev, initial=0.0)
            total_int = cum_int[-1]
            cum_int_at_a = cum_int[::-1] # Back to original order
            integral_a_to_1 = total_int - cum_int_at_a
            D_model = np.exp(-integral_a_to_1)
            
            # Errors
            rel_err_f = np.abs(f_model - f_ref) / np.abs(f_ref)
            rel_err_D = np.abs(D_model - D_ref) / np.abs(D_ref)
            
            max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))
            
        return max_err + penalty

    # Optimize
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    return res.x

def calc_reference_fast(om0, w0, wa, a_vals):
    # Returns f_ref, D_ref, om_m_a, u, a_hp_h
    # Integrate f ODE
    a_int = np.linspace(1.0, 0.1, 50)
    f_int = np.zeros_like(a_int)
    
    def get_params(a):
        u = 1.0 + w0 + wa * (1.0 - a)
        om_m = om0 * a**(-3)
        om_l = (1.0 - om0) * a**(-3 * u)
        h2 = om_m + om_l
        om_m_a = om_m / h2
        dh2_da = -3.0 * om_m / a - 3.0 * om_l / a * (u - wa)
        a_hp_h = 0.5 * a * dh2_da / h2
        return om_m_a, a_hp_h, u

    om_m_1, _, _ = get_params(1.0)
    f_int[0] = om_m_1**0.55
    
    for i in range(len(a_int) - 1):
        a_curr = a_int[i]
        a_next = a_int[i+1]
        da = a_next - a_curr
        
        om_m_a, a_hp_h, u = get_params(a_curr)
        f_curr = f_int[i]
        
        def df_da(a, f):
            om_m_a, a_hp_h, u = get_params(a)
            return (1.5 * om_m_a - f**2 - 0.5 * f - a_hp_h * f) / a
        
        k1 = df_da(a_curr, f_curr)
        k2 = df_da(a_curr + 0.5*da, f_curr + 0.5*da*k1)
        k3 = df_da(a_curr + 0.5*da, f_curr + 0.5*da*k2)
        k4 = df_da(a_curr + da, f_curr + da*k3)
        f_int[i+1] = f_curr + da/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        
    f_ref = np.interp(a_vals, a_int[::-1], f_int[::-1])
    
    # D_ref
    integrand = f_int * (1.0 / a_int)
    a_rev = a_int[::-1]
    integrand_rev = integrand[::-1]
    from scipy.integrate import cumulative_trapezoid
    cum_int = cumulative_trapezoid(integrand_rev, a_rev, initial=0.0)
    total_int = cum_int[-1]
    cum_int_at_a = np.interp(a_vals, a_rev, cum_int)
    integral_a_to_1 = total_int - cum_int_at_a
    D_ref = np.exp(-integral_a_to_1)
    
    # Grid params
    om_m_a_arr = np.zeros_like(a_vals)
    u_arr = np.zeros_like(a_vals)
    a_hp_h_arr = np.zeros_like(a_vals)
    for i, a in enumerate(a_vals):
        om_m_a_arr[i], a_hp_h_arr[i], u_arr[i] = get_params(a)
        
    return f_ref, D_ref, om_m_a_arr, u_arr, a_hp_h_arr
