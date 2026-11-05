import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        a = np.asarray(a, dtype=float)
        # Handle w0 = -1 case (Lambda CDM)
        if abs(w0 + 1.0) < 1e-8:
            if abs(wa) < 1e-8:
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
            else:
                # Limiting form for w0 -> -1, wa != 0
                # E^2 = Om a^-3 + (1-Om) * exp(3*wa*(1-a)/(1+w0)) * a^(-3(1+w0+wa))
                # As w0 -> -1, (1+w0) -> 0.
                # term = (1-Om) * a^(-3*wa) * exp(3*wa*(1-a)/(1+w0))
                # This is numerically tricky. 
                # Standard approximation: use a small epsilon for w0.
                eps = 1e-8
                w0_eff = -1.0 + eps
                exponent = -3.0 * (1.0 + w0_eff + wa)
                exp_term = 3.0 * wa * (1.0 - a) / (1.0 + w0_eff)
                exp_term = np.clip(exp_term, -500.0, 500.0)
                term2 = (1.0 - Om) * a**exponent * np.exp(exp_term)
                return np.sqrt(Om * a**(-3.0) + term2)
        else:
            exponent = -3.0 * (1.0 + w0 + wa)
            exp_term = 3.0 * wa * (1.0 - a) / (1.0 + w0)
            exp_term = np.clip(exp_term, -500.0, 500.0)
            term2 = (1.0 - Om) * a**exponent * np.exp(exp_term)
            return np.sqrt(Om * a**(-3.0) + term2)

    # Diverse training cosmologies
    # Grid of Om, w0, wa
    Om_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.3, 0.0, 0.3]
    
    models = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append({'Om': Om, 'w0': w0, 'wa': wa})
                
    # Add edge cases
    models.append({'Om': 0.15, 'w0': -0.7, 'wa': -0.3})
    models.append({'Om': 0.45, 'w0': -1.3, 'wa': 0.4})
    models.append({'Om': 0.3, 'w0': -1.0, 'wa': 0.0}) # LCDM
    
    redshifts = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    
    n_models = len(models)
    n_z = len(redshifts)
    
    # Pre-compute true integrals
    true_integrals = np.zeros((n_models, n_z))
    for i, m in enumerate(models):
        Om, w0, wa = m['Om'], m['w0'], m['wa']
        for j, z in enumerate(redshifts):
            def integrand(xi, Om=Om, w0=w0, wa=wa, z=z):
                a = 1.0 / (1.0 + z * xi)
                return 1.0 / E(a, Om, w0, wa)
            val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_integrals[i, j] = val
            
    true_flat = true_integrals.flatten()
    
    # Parameterization:
    # u: 8 params for nodes via sigmoid
    # v: 8 params for weights via softmax
    # Nodes: sort(sigmoid(u))
    # Weights: softmax(v)
    
    def decode_params(params):
        u = params[:8]
        v = params[8:]
        
        u_sorted = np.sort(u)
        nodes = 1.0 / (1.0 + np.exp(-u_sorted))
        # Clip to avoid exact 0 or 1
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        weights = exp_v / np.sum(exp_v)
        
        return nodes, weights

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes, weights = decode_params(params)
        
        # Check for degenerate nodes (too close)
        # If nodes are identical, quadrature is bad.
        # The sorting ensures order, but if u values are very close, nodes are close.
        # The error metric will naturally penalize this if the integrand is sensitive.
        
        # Vectorized computation
        # nodes: (8,)
        # redshifts: (n_z,)
        # a_vals: (8, n_z)
        a_vals = 1.0 / (1.0 + np.outer(nodes, redshifts))
        
        # Compute E for all models
        # We need E(a_vals) for each model.
        # E depends on Om, w0, wa.
        # We can loop over models but vectorize over nodes and redshifts.
        
        # Pre-allocate array for integrands: (n_models, 8, n_z)
        integrands = np.zeros((n_models, 8, n_z))
        
        for i, m in enumerate(models):
            Om, w0, wa = m['Om'], m['w0'], m['wa']
            e_vals = E(a_vals, Om, w0, wa)
            integrands[i] = 1.0 / e_vals
            
        # integrands shape: (n_models, 8, n_z)
        # weights shape: (8,)
        # Quadrature sum for each (model, z) is dot(weights, integrands[i, :, j])
        # We can reshape integrands to (n_models * n_z, 8) and multiply by weights
        integrands_flat = integrands.reshape(n_models * n_z, 8)
        quad_sums = integrands_flat @ weights
        
        # Compute relative errors
        rel_errors = np.abs(quad_sums - true_flat) / true_flat
        
        # Return max error (for DE) or smooth max (for gradient methods)
        # Since we use DE first, max is fine.
        # But L-BFGS-B needs smoothness.
        # We can use log-sum-exp with high beta for L-BFGS-B and max for DE.
        # However, a single objective is cleaner.
        # Let's use a smooth max with beta=50. This is differentiable and approximates max.
        beta = 50.0
        max_err = np.log(np.sum(np.exp(beta * rel_errors)) / len(rel_errors)) / beta
        
        return max_err

    # Initial guess: Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Encode GL
    u_init = np.log(nodes_gl / (1.0 - nodes_gl))
    v_init = np.log(weights_gl)
    x0 = np.concatenate([u_init, v_init])
    
    # Bounds for u and v
    # u: [-10, 10] covers sigmoid range well
    # v: [-10, 10] covers weight ratios well
    bounds = [(-10.0, 10.0)] * 16
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    # Phase 1: Differential Evolution
    # Use a small population and few iterations to save time, but enough to explore.
    # maxiter=20, popsize=15 -> ~300 evaluations.
    # Each evaluation is fast due to vectorization.
    try:
        def de_callback(xk, convergence):
            nonlocal best_x, best_val
            val = objective(xk)
            if val < best_val:
                best_val = val
                best_x = xk.copy()
            return time.time() > deadline

        de_res = differential_evolution(objective, bounds, maxiter=30, popsize=15, tol=1e-5, 
                                        callback=de_callback, seed=42, polish=False)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_x = de_res.x
    except Exception:
        pass

    # Phase 2: Local Refinement with L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Phase 3: Final Polish with Nelder-Mead (unconstrained, but our params are unconstrained)
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Decode final result
    nodes, weights = decode_params(best_x)
    
    # Ensure validity
    # Nodes should be strictly increasing in [0,1]
    # Weights should be positive and sum to 1
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(np.sum(weights) - 1.0) > 1e-12:
        # Fallback to GL
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    else:
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
        # Ensure strict positivity (already guaranteed by softmax, but just in case)
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
