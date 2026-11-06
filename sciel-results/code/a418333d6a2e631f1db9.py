import numpy as np
import scipy.optimize
import scipy.integrate
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Define representative cosmologies and redshifts
    # We need to cover the parameter space likely used in validation.
    # Let's sample a grid.
    om_m_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_values = [-1.5, -1.2, -1.0, -0.8, -0.5]
    wa_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    z_values = np.linspace(0.1, 5.0, 20)
    
    samples = []
    for om_m in om_m_values:
        for w0 in w0_values:
            for wa in wa_values:
                for z in z_values:
                    samples.append((om_m, w0, wa, z))
    
    # 2. Pre-compute exact integrals
    # Integrand for luminosity distance in terms of x = z'/z
    # I = z * integral_0^1 (1 + x*z) / E(a(x)) dx
    # a(x) = 1 / (1 + x*z)
    
    def E_of_a(a, om_m, w0, wa):
        # Flat universe
        om_r = 0.0001
        om_de0 = 1.0 - om_m - om_r
        # CPL dark energy density
        # rho_de(a) = rho_de0 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
        # H^2/H0^2 = om_m a^-3 + om_r a^-4 + om_de0 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
        if a <= 0:
            return np.inf
        term_m = om_m * a**(-3)
        term_r = om_r * a**(-4)
        term_de = om_de0 * a**(-3*(1+w0)) * np.exp(-3*wa*(1-a))
        return np.sqrt(term_m + term_r + term_de)
    
    exact_integrals = []
    integrands_cache = [] # Store function parameters for fast evaluation
    
    for om_m, w0, wa, z in samples:
        def integrand(x, om_m=om_m, w0=w0, wa=wa, z=z):
            a = 1.0 / (1.0 + x * z)
            E = E_of_a(a, om_m, w0, wa)
            return (1.0 + x * z) / E
        
        # Integrate from 0 to 1
        try:
            val, err = scipy.integrate.quad(integrand, 0, 1, limit=100)
            exact_integrals.append(val * z) # The integral is z * int ...
            # Store parameters for quick re-evaluation in optimization
            integrands_cache.append((om_m, w0, wa, z))
        except Exception:
            # If quad fails, skip this sample? Or use a fallback.
            # Quad should be robust for these smooth functions.
            pass

    # If we didn't get enough samples, reduce complexity
    if len(exact_integrals) < 10:
         # Fallback to simple rule
         n = 8
         return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    exact_integrals = np.array(exact_integrals)
    
    # 3. Define objective function
    # Variables: x_1..x_8, w_1..w_7. w_8 = 1 - sum(w_1..w_7)
    def objective(params):
        if time.time() > deadline:
            return 1e10 # Large penalty if timeout approached
        
        x = params[:8]
        w_first_7 = params[8:15]
        w_8 = 1.0 - np.sum(w_first_7)
        w = np.append(w_first_7, w_8)
        
        # Check constraints
        if np.any(w <= 0) or np.any(w_8 <= 0):
            return 1e10
        if np.any(x <= 0) or np.any(x >= 1):
            return 1e10
        # Check strictly increasing
        if np.any(np.diff(x) <= 0):
            return 1e10
            
        # Compute quadrature values
        # We need to evaluate f(x_i) for each sample
        # This is vectorized over samples if possible, but loops are okay if N is small
        # N_samples is ~ 5*5*5*20 = 2500. 8 nodes. 20k evaluations. Fast.
        
        max_rel_err = 0.0
        for i, (om_m, w0, wa, z) in enumerate(integrands_cache):
            # Calculate a_i for this sample's nodes
            a_i = 1.0 / (1.0 + x * z)
            # Calculate E(a_i)
            # Vectorized E calculation
            term_m = om_m * a_i**(-3)
            term_r = 0.0001 * a_i**(-4)
            om_de0 = 1.0 - om_m - 0.0001
            term_de = om_de0 * a_i**(-3*(1+w0)) * np.exp(-3*wa*(1-a_i))
            E_vals = np.sqrt(term_m + term_r + term_de)
            
            f_vals = (1.0 + x * z) / E_vals
            quad_val = np.dot(w, f_vals) * z
            
            rel_err = abs(quad_val - exact_integrals[i]) / abs(exact_integrals[i])
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                
        return max_rel_err

    # 4. Optimization
    # Initial guess: Midpoint rule
    x0 = (np.arange(8) + 0.5) / 8.0
    w0_init = np.full(8, 1.0/8.0)
    params0 = np.concatenate([x0, w0_init[:7]])
    
    # Bounds
    # x_i in [0.01, 0.99] to stay strictly inside
    # w_i in [1e-6, 0.999]
    bounds = []
    for _ in range(8):
        bounds.append((0.01, 0.99))
    for _ in range(7):
        bounds.append((1e-6, 0.999))
        
    # Use L-BFGS-B
    try:
        res = scipy.optimize.minimize(objective, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() > deadline:
             # Return best found so far? minimize might not return intermediate best easily.
             # We'll just use res if available.
             pass
             
        if res.success or res.fun < objective(params0):
            best_params = res.x
        else:
            best_params = params0
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        best_params = params0
        
    # 5. Extract final nodes and weights
    x_opt = best_params[:8]
    w_opt_first_7 = best_params[8:15]
    w_opt_8 = 1.0 - np.sum(w_opt_first_7)
    w_opt = np.append(w_opt_first_7, w_opt_8)
    
    # Ensure strict validity
    # Sort nodes? Optimization preserves order if started sorted and bounds are tight, 
    # but let's verify and sort if necessary (though weights must correspond).
    # If order changed, we must reorder weights.
    sort_idx = np.argsort(x_opt)
    x_opt = x_opt[sort_idx]
    w_opt = w_opt[sort_idx]
    
    # Final check
    if np.any(w_opt <= 0) or abs(np.sum(w_opt) - 1.0) > 1e-12 or np.any(x_opt <= 0) or np.any(x_opt >= 1):
         # Fallback
         n = 8
         return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
         
    return {"nodes": x_opt, "weights": w_opt}
