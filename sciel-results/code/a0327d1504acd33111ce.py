import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as spi

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Define a representative set of cosmologies
    # Parameters: Omega_m, w0, wa
    # We assume Omega_r is negligible for the quadrature shape sensitivity in this range,
    # or we can include a small fixed Omega_r. 
    # Standard: Omega_m ~ 0.3, Omega_L = 1 - Omega_m.
    
    omegas_m = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    w0s = np.array([-1.5, -1.2, -1.0, -0.8, -0.5])
    was = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    
    models = []
    for om in omegas_m:
        for w0 in w0s:
            for wa in was:
                models.append((om, w0, wa))
                
    def E_of_z(z, om_m, w0, wa):
        # E(z) = H(z)/H0
        # Omega_L = 1 - Omega_m
        om_l = 1.0 - om_m
        # Ignoring radiation for simplicity as it only affects very high z, 
        # and our z range is [0,1] (a from 1 to 0.5).
        # Actually, z in [0,1] is not very high. Radiation is small.
        term_m = om_m * (1.0 + z)**3
        term_l = om_l * (1.0 + z)**(3.0 * (1.0 + w0 + wa * z))
        return np.sqrt(term_m + term_l)

    def exact_integral(z, om_m, w0, wa):
        # Integrate 1/E(z') from 0 to z
        if z <= 1e-10:
            return 0.0
        res, err = spi.quad(lambda zp: 1.0 / E_of_z(zp, om_m, w0, wa), 0, z, limit=100)
        return res

    def compute_errors(nodes, weights, z_test):
        # nodes: u_i in [0,1]
        # weights: w_i
        # z_test: the redshift z for which we compute distance
        # Approximation: z * sum w_i / E(z*u_i)
        # Exact: integral
        # Relative error: |approx - exact| / exact
        
        errors = np.zeros(len(models))
        for i, (om_m, w0, wa) in enumerate(models):
            exact = exact_integral(z_test, om_m, w0, wa)
            if exact < 1e-10:
                continue
            # a_i = 1/(1+z*u_i)
            # E(a_i) = E(z*u_i)
            z_u = z_test * nodes
            E_vals = E_of_z(z_u, om_m, w0, wa)
            approx = z_test * np.sum(weights / E_vals)
            errors[i] = np.abs(approx - exact) / exact
        return errors

    # We want to minimize the worst-case error.
    # To make it differentiable, we use a smooth max (LogSumExp) of squared errors.
    # Or just minimize the sum of squared errors, hoping it correlates with max.
    # Given the time, minimizing the L_inf norm is hard to differentiate.
    # Let's minimize the sum of squared errors over a set of z values.
    # The error depends on z. The quadrature rule is for the integral from 0 to z.
    # The nodes u_i are fixed. The weights w_i are fixed.
    # The function being integrated changes with the model, but the integration variable transformation u = z'/z is universal.
    # So we are approximating I(f) = \int_0^1 f(u) du where f(u) = 1/E(z u).
    # The shape of f(u) depends on z and the model.
    # We should test a range of z values to ensure the rule works for all distances.
    # Let's pick a set of z values to test against.
    z_tests = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    def objective(params):
        # params: 8 nodes + 8 weights (unconstrained)
        # Transform to constrained
        # Nodes: use sigmoid to map to (0,1) and sort? 
        # Simpler: optimize nodes directly with bounds, and penalize if not sorted.
        # Weights: use softmax.
        
        u_params = params[:8]
        w_params = params[8:16]
        
        # Map u_params to nodes in (1e-6, 1-1e-6)
        # Using a cumulative sum of exp ensures positivity and sum, but we want nodes in [0,1].
        # Let's just use bounds in the optimizer and assume they stay sorted if initialized well.
        # Or, use a transformation: u_i = 1/(1+exp(-x_i)) but this doesn't guarantee order.
        # Let's use a simple mapping: u_i = (i-1)/7 + 0.1 * sin(x_i) ? No.
        
        # Let's use the cumulative softmax trick for nodes to ensure 0 < u_1 < ... < u_8 < 1
        # d_i = exp(u_params[i])
        # u_i = cumsum(d) / sum(d)
        # This maps R^8 to the simplex of ordered points in (0,1).
        d = np.exp(u_params - np.max(u_params)) # for stability
        d_sum = np.sum(d)
        nodes = np.cumsum(d) / d_sum
        # Ensure they are strictly inside (0,1)
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        # Enforce strict increase by adding a small epsilon if needed, though cumsum of positive is strictly increasing.
        
        # Weights: softmax
        w_exp = np.exp(w_params - np.max(w_params))
        weights = w_exp / np.sum(w_exp)
        
        # Calculate errors for all models and all z_tests
        max_err = 0.0
        sum_sq_err = 0.0
        count = 0
        for z in z_tests:
            errs = compute_errors(nodes, weights, z)
            # Use a high power to approximate max
            # Or just take max
            curr_max = np.max(errs)
            if curr_max > max_err:
                max_err = curr_max
            sum_sq_err += np.sum(errs**2)
            count += len(errs)
            
        # Return a combination of max error and mean error
        # To prioritize worst case:
        return max_err + 0.01 * (sum_sq_err / count)

    # Initial guess: Gaussian-Legendre nodes and weights
    # GL nodes are in [-1,1], map to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1) / 2
    gl_weights = gl_weights / 2 # since interval length is 1, weights are halved? 
    # Wait, \int_{-1}^1 f(x) dx = sum w_i f(x_i).
    # \int_0^1 f(u) du. Let u = (x+1)/2. du = dx/2.
    # \int_0^1 f(u) du = \int_{-1}^1 f((x+1)/2) dx/2 = sum (w_i/2) f((x_i+1)/2).
    # So weights are gl_weights / 2.
    # But our weights must sum to 1. Sum(gl_weights) = 2. Sum(gl_weights/2) = 1. Correct.
    
    # Invert the transformations for initial guess
    # Nodes: cumsum(d)/sum(d) = u. This is hard to invert directly.
    # Let's just start with uniform nodes and weights, or map GL nodes through the inverse?
    # Easier: Start with uniform nodes [1/16, 3/16, ...] and uniform weights [1/8].
    # And use the parameterization.
    # For uniform nodes u_i = i/8 (approx), the d_i would be roughly equal.
    # Let's just initialize params to 0.
    # If u_params = 0, d = 1, nodes = cumsum(1)/8 = [1/8, 2/8, ...].
    # If w_params = 0, weights = 1/8.
    
    x0 = np.zeros(16)
    
    # Optimization
    best_params = x0
    best_score = objective(x0)
    
    # Use L-BFGS-B
    # It might struggle with the non-smooth max.
    # Let's try Nelder-Mead if L-BFGS-B fails or is slow.
    # Given 16 dimensions, Nelder-Mead is feasible.
    
    try:
        res = so.minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline - 5:
            pass # Skip further refinement
        else:
            # Polish with L-BFGS-B?
            # L-BFGS-B needs gradients or finite differences. Finite diff is slow.
            # Let's just do a few more Nelder-Mead restarts if time permits.
            # Or a local search.
            
            # Try a few random restarts
            for i in range(5):
                if time.time() > deadline - 5:
                    break
                x_rand = np.random.normal(0, 1, 16)
                res_r = so.minimize(objective, x_rand, method='Nelder-Mead', 
                                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res_r.fun < best_score:
                    best_params = res_r.x
                    best_score = res_r.fun
                    
    except Exception:
        pass

    # Extract final nodes and weights
    u_params = best_params[:8]
    w_params = best_params[8:16]
    
    d = np.exp(u_params - np.max(u_params))
    d_sum = np.sum(d)
    nodes = np.cumsum(d) / d_sum
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    
    w_exp = np.exp(w_params - np.max(w_params))
    weights = w_exp / np.sum(w_exp)
    
    # Final check: weights sum to 1
    # They do by construction.
    
    return {"nodes": nodes, "weights": weights}
