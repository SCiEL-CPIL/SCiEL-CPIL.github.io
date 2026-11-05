"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    The rule is designed to minimize worst-case relative error across a diverse set of
    CPL (Chevallier-Polarski-Linder) dark energy models.
    
    Returns:
        dict: {'nodes': length-8 array, 'weights': length-8 array}
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin before 90s limit
    
    # Define a diverse set of CPL models for training
    # w0, wa pairs covering reasonable cosmological parameter space
    training_models = [
        (0.0, 0.0),    # Lambda CDM
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, -0.5),  # Quintessence-like
        (-1.5, 1.0),   # Phantom-like
        (-0.9, -0.1),  # Mildly evolving
        (-1.1, 0.3),   # Another variant
        (-0.7, -0.7),  # Stronger quintessence
        (-1.3, 0.5),   # Phantom
        (-0.8, 0.2),   # Moderate
        (-1.2, -0.2),  # Another
    ]
    
    def E_a(a, w0, wa):
        """H(a)/H0 for flat CPL model, where a = 1/(1+z)"""
        return (a ** (-3 * (1 + w0 + wa))) * ((1 + w0 * (1 - a) + wa * (1 - a) ** 2) ** (wa / (1 + w0)))
    
    def integrand(z, w0, wa):
        """Integrand: z * (1+z) / E(a) where a = 1/(1+z)"""
        a = 1.0 / (1.0 + z)
        return z * (1.0 + z) / E_a(a, w0, wa)
    
    def true_integral(z_max, w0, wa, n_points=1000):
        """Compute reference integral using adaptive quadrature"""
        from scipy.integrate import quad
        result, _ = quad(lambda z: integrand(z, w0, wa), 0, z_max, limit=200)
        return result
    
    def quadrature_error(nodes, weights, z_max, w0, wa):
        """Compute relative error of quadrature for given model and z_max"""
        a = 1.0 / (1.0 + z_max * nodes)
        E_vals = np.array([E_a(ai, w0, wa) for ai in a])
        quad_result = z_max * np.sum(weights / E_vals)
        true_val = true_integral(z_max, w0, wa)
        if abs(true_val) < 1e-15:
            return 0.0
        return abs(quad_result - true_val) / abs(true_val) * 100.0
    
    def objective(params):
        """
        Objective function: worst-case relative error across training set.
        params: [n0, n1, ..., n7, w0, w1, ..., w7] (16 values)
        """
        nodes = params[:8]
        weights = params[8:]
        
        # Check validity
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if not np.all(np.diff(nodes) > 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e10
        
        # Compute worst-case error over a range of z_max values
        z_max_values = [0.5, 1.0, 2.0, 3.0, 5.0]
        worst_err = 0.0
        for z_max in z_max_values:
            for w0, wa in training_models:
                err = quadrature_error(nodes, weights, z_max, w0, wa)
                worst_err = max(worst_err, err)
        return worst_err
    
    # Initial guess: equal-spaced nodes, equal weights
    n = 8
    init_nodes = (np.arange(n) + 0.5) / n
    init_weights = np.full(n, 1.0 / n)
    init_params = np.concatenate([init_nodes, init_weights])
    
    # Use differential evolution for global optimization
    from scipy.optimize import differential_evolution
    
    # Bounds: nodes in (0,1) with strict ordering, weights positive summing to 1
    # For nodes: 0 < n0 < n1 < ... < n7 < 1
    # We'll use a transformation to handle ordering constraint
    
    def transform_params(raw_params):
        """Transform raw parameters to valid nodes and weights"""
        # Nodes: use cumulative sum of positive values to ensure ordering
        raw_nodes = raw_params[:8]
        # Map to (0,1) with ordering using softmax-like or cumulative approach
        # Use: n_i = sum_{j=0}^{i} exp(raw_j) / sum_{k=0}^{7} exp(raw_k) * 0.9 + 0.05
        # This ensures 0.05 < n0 < n1 < ... < n7 < 0.95
        
        # Simpler: use order statistics of uniform random
        # n_i = (i + 0.5 + raw_i * 0.3) / 8, clamped
        nodes = np.zeros(8)
        for i in range(8):
            nodes[i] = (i + 0.5 + raw_nodes[i] * 0.4) / 8.0
        nodes = np.clip(nodes, 0.01, 0.99)
        # Ensure strict ordering
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 0.001
        nodes = np.clip(nodes, 0.01, 0.99)
        
        # Weights: use softmax to ensure positivity, then normalize
        raw_weights = raw_params[8:]
        weights = np.exp(raw_weights - np.max(raw_weights))
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    def objective_transformed(raw_params):
        """Objective with parameter transformation"""
        nodes, weights = transform_params(raw_params)
        return objective(np.concatenate([nodes, weights]))
    
    # Bounds for raw parameters
    bounds = [(-1.0, 1.0)] * 8 + [(-2.0, 2.0)] * 8
    
    # Run differential evolution with time monitoring
    best_result = None
    best_score = float('inf')
    
    try:
        result = differential_evolution(
            objective_transformed,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False
        )
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        # Return best found so far or initial guess
        if best_result is not None:
            nodes, weights = transform_params(best_result)
            # Final validation
            if np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0) and np.all(weights > 0):
                weights = weights / np.sum(weights)
                return {"nodes": nodes, "weights": weights}
        return {"nodes": init_nodes, "weights": init_weights}
    
    # Refine with local optimization if time allows
    if best_result is not None and time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        def objective_refined(params):
            nodes = params[:8]
            weights = params[8:]
            if np.any(nodes <= 0) or np.any(nodes >= 1) or not np.all(np.diff(nodes) > 0):
                return 1e10
            if np.any(weights <= 0):
                return 1e10
            if abs(np.sum(weights) - 1.0) > 1e-8:
                return 1e10
            return objective(np.concatenate([nodes, weights]))
        
        # Use transformed parameters as starting point
        init_nodes_t, init_weights_t = transform_params(best_result)
        init_params_t = np.concatenate([init_nodes_t, init_weights_t])
        
        try:
            refined = minimize(
                objective_refined,
                init_params_t,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if refined.fun < best_score:
                best_score = refined.fun
                best_result = refined.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    if best_result is not None:
        nodes, weights = transform_params(best_result)
    else:
        nodes, weights = init_nodes, init_weights
    
    # Final cleanup: ensure strict validity
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
