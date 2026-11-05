"""
Optimized 8-node positive quadrature rule for luminosity distance.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _E(a, w0, wa, w):
    """E(a) = H(a)/H0 for CPL dark energy model in flat universe.
    a = 1/(1+z), so (1-a) = z/(1+z) = z * a
    E(a)^2 = (1-a)^(-3(1+w0+w)) * a^(-3(1+w0))
    """
    a = np.asarray(a, dtype=np.float64)
    one_minus_a = 1.0 - a
    # Handle a close to 1 (z close to 0)
    eps = 1e-15
    one_minus_a = np.maximum(one_minus_a, eps)
    a_safe = np.maximum(a, eps)
    
    return np.power(one_minus_a, -3.0*(1.0 + w0 + w)) * np.power(a_safe, -3.0*(1.0 + w0))


def _integrand_z(z, w0, wa, w):
    """Integrand in z-space: z / E(z) where E(z) uses a = 1/(1+z)"""
    z = np.asarray(z, dtype=np.float64)
    a = 1.0 / (1.0 + z)
    return z / _E(a, w0, wa, w)


def _true_integral(z_max, w0, wa, w, n_sub=20000):
    """High-accuracy reference integral using adaptive Simpson or dense trapezoid."""
    z = np.linspace(0, z_max, n_sub + 1)
    f = _integrand_z(z, w0, wa, w)
    # Simpson's rule for high accuracy
    if n_sub % 2 == 1:
        n_sub -= 1
        z = np.linspace(0, z_max, n_sub + 1)
        f = _integrand_z(z, w0, wa, w)
    h = z_max / n_sub
    # Composite Simpson
    result = h / 3.0 * (f[0] + f[-1] + 4.0 * np.sum(f[1:-1:2]) + 2.0 * np.sum(f[2:-1:2]))
    return result


def _quadrature_error(z_max, nodes, weights, w0, wa, w):
    """Relative error of the quadrature rule vs true integral."""
    a = 1.0 / (1.0 + z_max * nodes)
    E_vals = _E(a, w0, wa, w)
    quad_val = z_max * np.sum(weights / E_vals)
    
    true_val = _true_integral(z_max, w0, wa, w)
    
    if true_val < 1e-15:
        return 0.0
    return abs(quad_val - true_val) / true_val


def _worst_error(nodes, weights, models, z_max_list):
    """Worst relative error across all models and z values."""
    max_err = 0.0
    for (w0, wa, w) in models:
        for z_max in z_max_list:
            err = _quadrature_error(z_max, nodes, weights, w0, wa, w)
            max_err = max(max_err, err)
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Training models: diverse CPL parameters
    models = [
        (0.0, 0.0, 0.0),    # LambdaCDM
        (-0.5, 0.5, 0.0),   # Quintessence-like
        (-1.0, 0.0, 0.0),   # Cosmological constant
        (-0.5, -0.5, 0.0),  # Phantom-like
        (-1.5, 1.0, 0.0),   # Strongly varying
        (0.0, -1.0, 0.0),   # Different behavior
        (-0.8, 0.3, 0.0),   # Intermediate
        (-1.2, -0.3, 0.0),  # Another variant
    ]
    
    z_max_list = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Initial guess: Gauss-Legendre-like nodes with equal weights
    # Start with a reasonable spread
    x0_nodes = np.array([0.05, 0.15, 0.3, 0.5, 0.65, 0.8, 0.9, 0.98])
    x0_weights = np.ones(8) / 8.0
    
    x0 = np.concatenate([x0_nodes, x0_weights])
    
    best_x = x0.copy()
    best_err = _worst_error(x0_nodes, x0_weights, models, z_max_list)
    
    def objective(x):
        # x = [n1..n8, w1..w8]
        nodes = x[:8]
        weights = x[8:]
        
        # Constraints: nodes strictly increasing in [0,1], weights positive, sum to 1
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e6
        for i in range(7):
            if nodes[i+1] - nodes[i] < 1e-8:
                return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e6
        
        return _worst_error(nodes, weights, models, z_max_list)
    
    # Use SLSQP for constrained optimization
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
        {'type': 'ineq', 'fun': lambda x: x[0] - 1e-6},  # n1 > 0
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[7]},    # n8 < 1
    ]
    # Strictly increasing
    for i in range(7):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 1e-6})
    # Positive weights
    for i in range(8):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[8+i] - 1e-10})
    
    bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(1e-10, None)] * 8
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 5
    
    for s in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        if s == 0:
            x_start = x0.copy()
        else:
            # Perturb initial guess
            nodes_pert = x0_nodes + np.random.randn(8) * 0.05
            nodes_pert = np.sort(np.clip(nodes_pert, 1e-4, 1.0 - 1e-4))
            # Ensure strict increase
            for i in range(7):
                if nodes_pert[i+1] - nodes_pert[i] < 1e-5:
                    nodes_pert[i+1] = nodes_pert[i] + 1e-4
            weights_pert = np.abs(np.random.randn(8)) + 0.1
            weights_pert /= np.sum(weights_pert)
            x_start = np.concatenate([nodes_pert, weights_pert])
        
        try:
            res = minimize(objective, x_start, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success and res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30.0:
            break
    
    # Polish with more iterations
    if time.time() < deadline - 15.0:
        try:
            res = minimize(objective, best_x, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Final cleanup: ensure constraints are met
    nodes = np.sort(np.clip(nodes, 1e-8, 1.0 - 1e-8))
    # Ensure strict increase
    for i in range(7):
        if nodes[i+1] - nodes[i] < 1e-8:
            nodes[i+1] = nodes[i] + 1e-7
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    
    weights = np.maximum(weights, 1e-12)
    weights /= np.sum(weights)
    
    # Verify sum is exactly 1 within tolerance
    adjustment = 1.0 - np.sum(weights)
    weights[0] += adjustment
    if weights[0] < 1e-12:
        # Redistribute
        weights = np.maximum(weights, 1e-12)
        weights /= np.sum(weights)
    
    return {"nodes": nodes.astype(np.float64), "weights": weights.astype(np.float64)}
# EVOLVE-BLOCK-END
