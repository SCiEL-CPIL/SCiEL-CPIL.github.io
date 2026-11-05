import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        a = np.asarray(a, dtype=float)
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

    # Diverse training cosmologies covering parameter space
    # Using a grid to ensure broad coverage
    Om_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.3, 0.0, 0.3]
    
    models = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append({'Om': Om, 'w0': w0, 'wa': wa})
                
    # Add some specific edge cases or interesting ones
    models.append({'Om': 0.15, 'w0': -0.7, 'wa': -0.3})
    models.append({'Om': 0.45, 'w0': -1.3, 'wa': 0.4})
    
    redshifts = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    
    # Pre-compute true integrals using quad
    true_integrals = np.zeros((len(models), len(redshifts)))
    for i, m in enumerate(models):
        Om, w0, wa = m['Om'], m['w0'], m['wa']
        for j, z in enumerate(redshifts):
            def integrand(xi, Om=Om, w0=w0, wa=wa, z=z):
                a = 1.0 / (1.0 + z * xi)
                return 1.0 / E(a, Om, w0, wa)
            val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_integrals[i, j] = val

    n_models = len(models)
    n_z = len(redshifts)

    # Parameterization:
    # Nodes: x_i = sigmoid(u_i). To ensure strict ordering, we sort u_i and use the sorted values for nodes.
    # However, weights must correspond to the nodes. If we sort nodes, we must effectively permute weights.
    # A simpler approach for optimization:
    # Let u be 8 parameters. Sort u -> u_sorted. Nodes = sigmoid(u_sorted).
    # Weights are determined by 8 parameters v. w_i = exp(v_i) / sum(exp(v_j)).
    # The issue is that if we sort nodes, the index i in weights must correspond to the i-th smallest node.
    # So we define the quadrature as sum_{i=1..8} w_i * f(x_i) where x_1 < ... < x_8.
    # This means the "identity" of the weight is tied to the rank of the node.
    # So we can just optimize u (unsorted) and v.
    # In the objective, we sort u to get node ranks, and use v in the original order?
    # No, if we sort u, the nodes change order. The weights v are just a vector.
    # If we swap u_i and u_j, the nodes swap. The weights v_i and v_j stay put.
    # This effectively allows the optimizer to "assign" weights to node positions.
    # So:
    # 1. Compute u_sorted = sort(u).
    # 2. Compute nodes = sigmoid(u_sorted).
    # 3. Compute weights from v.
    # 4. The quadrature uses these nodes and weights.
    # This is valid because the set of (node, weight) pairs is what matters.
    # Wait, if I have nodes [0.2, 0.5] and weights [0.1, 0.9], the sum is 0.1*f(0.2) + 0.9*f(0.5).
    # If I swap u so nodes become [0.5, 0.2] (but we sort nodes to [0.2, 0.5]), the nodes are the same.
    # The weights are still [0.1, 0.9]. So the sum is the same.
    # So the mapping is:
    # - u defines the set of node positions (via sorting and sigmoid).
    # - v defines the set of weights.
    # - The pairing is by index after sorting nodes.
    # This is a valid representation.

    def decode_params(params):
        u = params[:8]
        v = params[8:]
        
        # Sort u to get ordered nodes
        u_sorted = np.sort(u)
        nodes = 1.0 / (1.0 + np.exp(-u_sorted))
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
        for i in range(n_models):
            Om, w0, wa = models[i]['Om'], models[i]['w0'], models[i]['wa']
            for j in range(n_z):
                z = redshifts[j]
                a_vals = 1.0 / (1.0 + z * nodes)
                e_vals = E(a_vals, Om, w0, wa)
                q_val = np.dot(weights, 1.0 / e_vals)
                true_val = true_integrals[i, j]
                rel_err = abs(q_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    # Initial guess: Gauss-Legendre nodes and weights
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Encode GL nodes and weights
    # u = logit(nodes)
    u_init = np.log(nodes_gl / (1.0 - nodes_gl))
    # v = log(weights)
    v_init = np.log(weights_gl)
    
    x0 = np.concatenate([u_init, v_init])
    
    # Bounds for u and v
    # u: sigmoid maps R to (0,1). Let's use [-10, 10]
    # v: exp maps R to (0, inf). Let's use [-10, 10]
    bounds = [(-10.0, 10.0)] * 16
    
    best_x = x0
    best_val = objective(best_x)
    
    # Differential Evolution for global search
    # This is robust to non-convexity
    try:
        def de_callback(xk, convergence):
            nonlocal best_x, best_val
            val = objective(xk)
            if val < best_val:
                best_val = val
                best_x = xk.copy()
            return time.time() > deadline

        de_res = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-6, 
                                        callback=de_callback, seed=42, polish=False)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_x = de_res.x
    except Exception:
        pass

    # Local refinement with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            # Use L-BFGS-B on the unbounded parameters (u, v)
            # Since constraints are handled by parameterization, we don't need bounds for L-BFGS-B
            # but we can provide them to keep values reasonable.
            res = minimize(objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with Nelder-Mead (often good for final steps)
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Decode final result
    nodes, weights = decode_params(best_x)
    
    # Ensure nodes are strictly increasing
    # They should be due to sorting in decode, but just in case
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx] # This is wrong if weights were tied to unsorted u indices
    # Wait, in decode_params, nodes are sorted. Weights are NOT sorted.
    # The quadrature sum is sum_{i} w_i f(x_i).
    # If I sort nodes, I must keep the weights associated with the original index i?
    # No, the quadrature rule is defined by the set of pairs (x_i, w_i).
    # If I have pairs (x_1, w_1), (x_2, w_2), ...
    # And I sort nodes to x_1' < x_2' < ...
    # The weights must follow the nodes.
    # In decode_params:
    # u_sorted = sort(u). nodes = sigmoid(u_sorted).
    # weights = softmax(v).
    # The i-th node is nodes[i]. The i-th weight is weights[i].
    # This implies that the weight weights[i] is paired with the i-th smallest node.
    # This is a valid interpretation. The optimizer chooses v such that the i-th smallest node gets weight v_i.
    # So if I sort nodes again, I don't need to sort weights, because they are already associated by index.
    # However, if I did something else...
    # Let's stick to the decode_params logic.
    # nodes = sigmoid(sort(u))
    # weights = softmax(v)
    # The pair is (nodes[i], weights[i]).
    # So I should NOT sort weights separately if I already sorted nodes in decode.
    # But wait, if I call decode_params, nodes are sorted.
    # If I then do sort_idx = argsort(nodes), it will be [0, 1, ..., 7].
    # So nodes[sort_idx] = nodes. weights[sort_idx] = weights.
    # So it's fine.
    
    # Check validity
    if (not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or 
        abs(np.sum(weights) - 1.0) > 1e-8 or nodes[0] <= 0 or nodes[-1] >= 1):
        # Fallback to Gauss-Legendre
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    else:
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
        # Ensure strict positivity
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
