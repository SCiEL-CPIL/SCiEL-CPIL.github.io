import numpy as np
from scipy.optimize import minimize
import time

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Define diverse CPL models for training
    # w0, wa pairs - cover broad range including extremes
    w0_vals = [-2.0, -1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.3, 0.5, 1.0]
    wa_vals = [-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    models = []
    for w0 in w0_vals:
        for wa in wa_vals:
            models.append((w0, wa))
    # Add LambdaCDM
    models.append((-1.0, 0.0))
    # Add some physically motivated extremes
    models.append((-1.5, 1.0))
    models.append((-0.5, -1.0))
    models.append((0.0, 0.0))  # empty universe
    models.append((-1.0, -1.0))
    models.append((-1.0, 1.0))

    Om_m = 0.3
    Om_de = 1.0 - Om_m

    def E_of_a_vec(z, w0, wa):
        """Vectorized E(a) for array of z values"""
        w = w0 + wa * z
        E2 = (1.0 + z)**3 + Om_de * (1.0 + z)**(3.0 * (1.0 + w))
        return np.sqrt(np.maximum(E2, 1e-30))

    def reference_integral(w0, wa, n_points=50000):
        """High-accuracy reference using Simpson's rule"""
        z = np.linspace(0, 1, n_points)
        E = E_of_a_vec(z, w0, wa)
        f = (1.0 + z) * z / E
        # Simpson's rule
        if (n_points - 1) % 2 != 0:
            n_points -= 1
            z = np.linspace(0, 1, n_points)
            E = E_of_a_vec(z, w0, wa)
            f = (1.0 + z) * z / E
        h = 1.0 / (n_points - 1)
        result = h/3.0 * (f[0] + f[-1] + 4*np.sum(f[1:-1:2]) + 2*np.sum(f[2:-1:2]))
        return result

    # Precompute reference integrals
    refs = np.array([reference_integral(w0, wa) for w0, wa in models])

    def quadrature_value(nodes, weights, w0, wa):
        E = E_of_a_vec(nodes, w0, wa)
        f = (1.0 + nodes) * nodes / E
        return np.dot(weights, f)

    def compute_errors(nodes, weights):
        """Compute relative errors for all models, return max"""
        max_err = 0.0
        for i, (w0, wa) in enumerate(models):
            q_val = quadrature_value(nodes, weights, w0, wa)
            ref_val = refs[i]
            if ref_val > 1e-10:
                rel_err = abs(q_val - ref_val) / abs(ref_val)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    def param_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights"""
        # Nodes: use cumulative log increments
        # node[0] = sigmoid(params[0]) * 0.15 + 0.02
        # node[i] = node[i-1] + exp(params[i]) * 0.05
        # This guarantees strict increasing order
        p = params[:8]
        w_p = params[8:16]
        
        # Node parameterization: ensure strictly increasing in (0,1)
        # Use: x0 = sigmoid(p0)*0.12 + 0.01
        #      xi = x_{i-1} + exp(p_i - 5) * 0.08
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-p[0])) * 0.12 + 0.01
        for i in range(1, 8):
            increment = np.exp(p[i] - 4.0) * 0.1
            nodes[i] = nodes[i-1] + increment
        # If nodes go out of range, scale
        if nodes[-1] >= 0.99:
            scale = 0.98 / nodes[-1]
            nodes = nodes * scale
            nodes[0] = max(nodes[0], 0.005)
        # Ensure all in (0, 1)
        nodes = np.clip(nodes, 0.001, 0.999)
        # Ensure strict ordering
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 0.001
        nodes = np.clip(nodes, 0.001, 0.999)
        
        # Weights: softmax
        w_raw = np.exp(w_p - np.max(w_p))
        weights = w_raw / np.sum(w_raw)
        
        return nodes, weights

    def objective(params):
        nodes, weights = param_to_nodes_weights(params)
        return compute_errors(nodes, weights)

    # Initial guess: roughly uniform nodes, equal weights
    x0 = np.zeros(16)
    # Set nodes to approximately uniform: node_i ≈ (i+0.5)/8
    # Invert: nodes[0] = sigmoid(p0)*0.12+0.01 ≈ 0.0625 => sigmoid(p0) ≈ 0.4375 => p0 ≈ -0.25
    # nodes[i] - nodes[i-1] ≈ 0.125 => exp(p_i-4)*0.1 ≈ 0.125 => p_i-4 ≈ ln(1.25) ≈ 0.223 => p_i ≈ 4.223
    x0[0] = -0.25
    for i in range(1, 8):
        x0[i] = 4.0 + np.log(0.125/0.1)
    x0[8:16] = 0.0  # equal weights

    best_params = x0.copy()
    best_err = objective(x0)

    # Optimization strategy 1: L-BFGS-B with multiple starts
    np.random.seed(42)
    starts = [x0]
    for _ in range(8):
        starts.append(x0 + 0.3 * np.random.randn(16))

    for start in starts:
        if time.time() - t0 > deadline - 15:
            break
        try:
            res = minimize(objective, start, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass

    # Optimization strategy 2: Nelder-Mead for local refinement
    if time.time() - t0 < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass

    # Optimization strategy 3: Powell for final polish
    if time.time() - t0 < deadline - 5:
        try:
            res = minimize(objective, best_params, method='Powell',
                          options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract final nodes and weights
    nodes, weights = param_to_nodes_weights(best_params)

    # Final validation
    weights = weights / np.sum(weights)
    assert np.all(np.diff(nodes) > 0), "Nodes not strictly increasing"
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes out of range"
    assert np.all(weights > 0), "Weights not positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights don't sum to 1"

    return {"nodes": nodes, "weights": weights}
