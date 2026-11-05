import numpy as np
import scipy.optimize
import scipy.integrate
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Surrogate grid of cosmologies
    # Om in [0.1, 0.4], w0 in [-1.2, 0.2], wa in [-0.5, 0.5]
    # We want to cover the range likely to be in the validation set.
    om_vals = np.linspace(0.1, 0.4, 5)
    w0_vals = np.linspace(-1.2, 0.2, 5)
    wa_vals = np.linspace(-0.5, 0.5, 5)
    
    cosmologies = []
    for om in om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om, w0, wa))
                
    # Scale factor grid
    a_vals = np.linspace(0.01, 1.0, 50)
    
    # Precompute reference solutions
    # D(a) and f(a) = d ln D / d ln a
    ref_D = []
    ref_f = []
    ref_om = []
    ref_q = []
    ref_u = []
    ref_a = []
    
    for om, w0, wa in cosmologies:
        # Compute H(a)
        # H^2 = H0^2 [ Om a^-3 + Oml a^-3(1+w0+wa/3) exp(-3 wa (1-a)) ]
        # For simplicity, we solve the ODE for D.
        # ODE: D'' + (3/2 + H'/H) D' - 3/2 Om(a) D = 0
        
        def h2(a):
            # Ignore radiation and curvature
            de_term = om * a**(-3)
            # Dark energy density ratio: 
            # rho_L(a) = rho_L0 * a^(-3(1+w0)) * exp(-3 wa (1-a))
            # Omega_L(a) = (1-om) * a^(-3(1+w0)) * exp(-3 wa (1-a)) / (a^(-3) + (1-om) a^(-3(1+w0)) exp(-3 wa (1-a)))
            # This is getting complicated. Let's just compute H'/H directly.
            # H^2 = H0^2 [ Om a^-3 + (1-Om) a^-3(1+w0) exp(-3 wa (1-a)) ]
            # Let X = Om a^-3, Y = (1-Om) a^-3(1+w0) exp(-3 wa (1-a))
            # H^2 = H0^2 (X + Y)
            # d ln H^2 / d ln a = (X' + Y') / (X + Y)
            # X' = -3 X
            # Y = (1-Om) a^(-3-3w0) exp(-3 wa + 3 wa a)
            # Y' = Y * (-3-3w0 + 3 wa)
            x = om * a**(-3)
            y = (1-om) * a**(-3*(1+w0)) * np.exp(-3*wa*(1-a))
            if x + y == 0: return 0
            dx = -3 * x
            dy = y * (-3*(1+w0) + 3*wa)
            return (dx + dy) / (x + y)
        
        def ode_funcs(t, y):
            a = t
            d, dp = y
            hprime_over_h = h2(a) / 2.0 # d ln H / d ln a = 1/2 d ln H^2 / d ln a
            # Wait, the ODE is in terms of a.
            # D'' + (3/2 + H'/H) D' - 3/2 Om(a) D = 0
            # Here ' denotes d/da.
            # H'/H in the ODE usually means (dH/da)/H.
            # My h2(a) calculation above was d ln H^2 / d ln a = a * (dH^2/da)/H^2 = 2 a (dH/da)/H.
            # So a * (H'/H) = h2(a)/2.
            # Thus H'/H = h2(a) / (2a).
            hp_h = h2(a) / (2 * a)
            
            om_a = om * a**(-3) / (om * a**(-3) + (1-om) * a**(-3*(1+w0)) * np.exp(-3*wa*(1-a)))
            
            d2 = - (1.5 + hp_h) * dp + 1.5 * om_a * d
            return [dp, d2]
        
        # Solve ODE from a=1 backwards to a=0.01
        # Initial conditions at a=1: D(1)=1, D'(1)=f(1)
        # f(1) approx 0.55 * Om(1)^0.55 ? No, f(1) = Om(1)^gamma.
        # We can estimate f(1) using the constant gamma 0.55.
        f1 = om ** 0.55
        sol = scipy.integrate.solve_ivp(
            ode_funcs, [1.0, 0.01], [1.0, f1], 
            method='LSODA', rtol=1e-8, atol=1e-10, dense_output=True
        )
        
        if not sol.success:
            continue
            
        # Evaluate on grid
        D_vals = sol.sol(a_vals)[0]
        Dp_vals = sol.sol(a_vals)[1]
        
        # f = d ln D / d ln a = a * D' / D
        f_vals = a_vals * Dp_vals / D_vals
        
        # Store
        ref_D.append(D_vals / D_vals[-1]) # Normalize D(1)=1
        ref_f.append(f_vals)
        ref_om.append(np.array([om * a**(-3) / (om * a**(-3) + (1-om) * a**(-3*(1+w0)) * np.exp(-3*wa*(1-a))) for a in a_vals]))
        ref_q.append(1 - ref_om[-1])
        ref_u.append(np.array([1 + w0 + wa*(1-a) for a in a_vals]))
        ref_a.append(a_vals)
        
    if not ref_D:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    # Flatten for vectorized operations in loss function
    ref_D_flat = np.concatenate(ref_D)
    ref_f_flat = np.concatenate(ref_f)
    ref_om_flat = np.concatenate(ref_om)
    ref_q_flat = np.concatenate(ref_q)
    ref_u_flat = np.concatenate(ref_u)
    ref_a_flat = np.concatenate(ref_a)
    
    n_points = len(ref_D_flat)
    
    def loss(params):
        if time.time() > deadline:
            return 1e10
            
        c0, c1, c2, c3, c4, c5 = params
        
        # Check coefficient bounds
        if np.any(np.abs(params) > 10):
            return 1e10
            
        q = ref_q_flat
        u = ref_u_flat
        a = ref_a_flat
        om = ref_om_flat
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*params[4]*a + c5*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10 + np.max(np.abs(gamma - 0.65))
            
        f_approx = om ** gamma
        
        # Calculate D_approx
        # d ln D / dx = f, x = ln a
        # We need to integrate from 1 to a.
        # Since we have discrete points, we can use cumulative integration.
        # But the points are not sorted by a in the flattened array? 
        # They are grouped by cosmology. 
        # It's safer to calculate D for each cosmology separately or ensure sorting.
        # Let's sort by a? No, the reference data is paired.
        # We can compute the integral numerically.
        # x = ln(a). dx = da/a.
        # ln D(a) - ln D(1) = - \int_a^1 f(x) dx/x  (since D(1)=1)
        # Or \int_1^a f(x) dx/x.
        
        # To avoid complexity of integration on flattened array, 
        # we can just use the f_approx error and a simple D approximation 
        # or compute D error properly.
        
        # Let's compute D error properly by iterating over cosmologies.
        # This might be slow in the inner loop.
        # Alternative: Use a simple trapezoidal rule on the flattened array? 
        # No, the a values repeat.
        
        # Let's optimize the loss function to be faster.
        # We can pre-sort the data by cosmology and a.
        # But for now, let's just compute f error and use a proxy for D error?
        # No, D error is significant.
        
        # Let's reconstruct the integration.
        # Since n_points is ~125*50 = 6250, it's small.
        # We can loop over cosmologies.
        
        max_err = 0.0
        for i in range(len(ref_D)):
            # Extract slice for this cosmology
            start = i * len(a_vals)
            end = (i+1) * len(a_vals)
            
            f_i = f_approx[start:end]
            a_i = ref_a_flat[start:end]
            D_ref_i = ref_D_flat[start:end]
            
            # Integrate f_i to get ln D
            # x = ln a
            x = np.log(a_i)
            # We want D(a) = exp( \int_1^a f(x') dx' )
            # Since a_i goes from 1.0 down to 0.01 in our ODE solve?
            # No, a_vals = np.linspace(0.01, 1.0, 50).
            # So a_i is increasing.
            # \int_1^a f dx = - \int_a^1 f dx.
            # Let's use cumulative trapezoidal integration.
            
            # Sort by x if necessary? a_vals is already sorted.
            # We need integral from 1 to a_i.
            # x is increasing. 
            # Let's compute cumulative integral from the start (a=0.01) and subtract?
            # Easier: compute integral from 1 to a_i.
            # Since 1 is the last point, we can integrate backwards or use scipy.
            
            # Simple trapezoidal rule for \int_1^{a_i} f dx
            # We can use np.cumtrapz on the reversed array?
            # Or just use a simple loop? No, vectorize.
            
            # Let's use scipy.integrate.cumulative_trapezoid
            # It computes \int_0^x. 
            # We want \int_1^x.
            # \int_1^x = \int_0^x - \int_0^1.
            
            # x = log(a). a ranges [0.01, 1]. x ranges [ln(0.01), 0].
            # f_i is defined on a_i.
            
            try:
                ln_D_rel = scipy.integrate.cumulative_trapezoid(f_i, x, initial=0)
                # ln_D_rel is \int_{x_0}^{x} f dx. x_0 = ln(0.01).
                # We want \int_{0}^{x} f dx (since x(1)=0).
                # \int_{0}^{x} = \int_{x_0}^{x} - \int_{x_0}^{0}.
                # \int_{x_0}^{0} is the last element of ln_D_rel.
                total_int = ln_D_rel[-1]
                ln_D_from_1 = ln_D_rel - total_int
                D_approx_i = np.exp(ln_D_from_1)
            except Exception:
                return 1e10
                
            # Relative errors
            err_f = np.abs(f_i - ref_f_flat[start:end]) / (np.abs(ref_f_flat[start:end]) + 1e-10)
            err_D = np.abs(D_approx_i - D_ref_i) / (np.abs(D_ref_i) + 1e-10)
            
            curr_max = np.max(np.concatenate([err_f, err_D]))
            if curr_max > max_err:
                max_err = curr_max
                
        return max_err

    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    best_params = x0
    best_loss = loss(x0)
    
    # Use Nelder-Mead for robustness with constraints handled by penalty
    try:
        res = scipy.optimize.minimize(
            loss, x0, method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
        )
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun
    except Exception:
        pass
        
    # Refine with L-BFGS-B if possible? 
    # Nelder-Mead is often better for small non-smooth problems.
    
    # Ensure final check
    if loss(best_params) < 1e9:
        return [float(x) for x in best_params]
    else:
        return x0

# EVOLVE-BLOCK-END
