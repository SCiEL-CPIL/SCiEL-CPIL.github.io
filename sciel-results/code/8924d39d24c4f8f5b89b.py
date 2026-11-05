import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    # 1. Define E(a) for CPL dark energy
    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        a = np.asarray(a)
        if abs(w0 + 1.0) < 1e-8:
            if abs(wa) < 1e-8:
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
            else:
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

    # 2. Define diverse training cosmologies
    # Cover a wide range of parameters to ensure universality
    Om_vals = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    wa_vals = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    
    models = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append({'Om': Om, 'w0': w0, 'wa': wa})
                
    # Redshifts to cover the integration range more densely
    redshifts = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    # Pre-compute true integrals
    true_integrals = np.zeros((len(models), len(redshifts)))
    for i, m in enumerate(models):
        Om, w0, wa = m['Om'], m['w0'], m['wa']
        for j, z in enumerate(redshifts):
            def integrand(xi):
                a = 1.0 / (1.0 + z * xi)
                return 1.0 / E(a, Om, w0, wa)
            val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_integrals[i, j] = val

    # 3. Parameterization
    # Nodes: x_i = sigmoid(y_i). To ensure x_1 < x_2 < ... < x_8, 
    # we can define x_i = sigmoid(cumsum(z_i)) where z_i are parameters? 
    # No, that doesn't guarantee strict ordering easily without care.
    # Better: Let nodes be determined by 7 positive gaps.
    # x_1 = sigmoid(g_0)
    # x_2 = x_1 + sigmoid(g_1) * (1 - x_1) ... this is complex.
    
    # Simpler robust parameterization for nodes in (0,1) sorted:
    # Use 8 parameters p_i. 
    # Sort p_i. Map sorted p_i to (0,1) via sigmoid.
    # But this loses the "which node is which" identity if we swap weights.
    # We need nodes and weights to be paired.
    
    # Alternative: Optimize 8 nodes n_i and 8 weights w_i directly but with soft constraints?
    # Or use the transformation:
    # n_i = sigmoid(u_i)
    # Sort u_i -> u_sorted. n_i = sigmoid(u_sorted[i]).
    # But weights must correspond to the original indices? No, weights are just a vector.
    # If we sort nodes, we must sort weights accordingly? 
    # Actually, the quadrature sum is sum w_i f(n_i). The order of pairs doesn't matter for the sum.
    # So we can just find 8 nodes and 8 weights.
    # To ensure nodes are in (0,1) and distinct, we can use:
    # u_i are free parameters.
    # n_i = sigmoid(u_i).
    # If two n_i are very close, it's bad, but the objective will penalize it if it hurts accuracy.
    # To ensure distinctness, we can add a small penalty for |n_i - n_j|.
    
    # Weights: w_i = exp(v_i) / sum(exp(v_j)). This ensures w_i > 0 and sum w_i = 1.
    
    def decode_params(params):
        u = params[:8]
        v = params[8:]
        nodes = 1.0 / (1.0 + np.exp(-u))
        # Ensure nodes are strictly inside (0,1)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Calculate weights
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        weights = exp_v / np.sum(exp_v)
        
        return nodes, weights

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes, weights = decode_params(params)
        
        # Check for duplicate nodes (should be rare with continuous params)
        # If nodes are too close, the quadrature is ill-conditioned.
        # But let's rely on the error metric.
        
        # Compute errors
        max_err = 0.0
        for i, m in enumerate(models):
            Om, w0, wa = m['Om'], m['w0'], m['wa']
            for j, z in enumerate(redshifts):
                a_vals = 1.0 / (1.0 + z * nodes)
                e_vals = E(a_vals, Om, w0, wa)
                q_val = np.dot(weights, 1.0 / e_vals)
                true_val = true_integrals[i, j]
                rel_err = abs(q_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    # 4. Optimization with multiple restarts
    # Initial guess: Gauss-Legendre nodes and weights
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Encode GL nodes and weights
    u_init = np.log(nodes_gl / (1.0 - nodes_gl))
    # Weights are proportional to exp(v). So v = log(weights) + const.
    v_init = np.log(weights_gl)
    
    x0 = np.concatenate([u_init, v_init])
    
    # Bounds for u and v
    # u: sigmoid maps R to (0,1). No strict bounds needed, but let's say [-10, 10]
    # v: exp maps R to (0, inf). No strict bounds needed, but let's say [-10, 10]
    bounds = [(-10.0, 10.0)] * 16
    
    best_x = x0
    best_val = objective(best_x)
    
    # Differential Evolution for global search with multiple restarts
    # This is robust to non-convexity
    rng = np.random.default_rng(42)
    for restart in range(3):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            de_seed = 42
        else:
            de_seed = 42 + restart * 1000
        
        try:
            de_res = differential_evolution(objective, bounds, maxiter=80, popsize=12, tol=1e-6, 
                                            callback=lambda xk, convergence: (time.time() > deadline - 10.0), 
                                            seed=de_seed, polish=False)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x
        except Exception:
            pass

    # Local refinement with L-BFGS-B
    if time.time() < deadline - 8.0:
        try:
            # Use L-BFGS-B on the unbounded parameters (u, v)
            # Since constraints are handled by parameterization, we don't need bounds for L-BFGS-B
            # but we can provide them to keep values reasonable.
            res = minimize(objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with Nelder-Mead (often good for final steps)
    if time.time() < deadline - 3.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # One more L-BFGS-B polish after Nelder-Mead
    if time.time() < deadline - 1.0:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Decode final result
    nodes, weights = decode_params(best_x)
    
    # Sort nodes and weights together? 
    # The problem statement says "Nodes must be strictly increasing".
    # Our parameterization doesn't enforce ordering of u, so nodes might not be sorted.
    # We must sort them.
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx]
    
    # Check validity
    if (not np.all(np.diff(nodes) > 1e-6) or not np.all(weights > 0) or 
        abs(np.sum(weights) - 1.0) > 1e-10 or nodes[0] <= 0 or nodes[-1] >= 1):
        # Fallback to GL
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    else:
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
