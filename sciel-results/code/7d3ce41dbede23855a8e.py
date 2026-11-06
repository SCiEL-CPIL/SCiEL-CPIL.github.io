"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    return abs(approx - exact) / exact * 100.0

def _gauss_legendre_nodes(n):
    """Generate Gauss-Legendre nodes and weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    w = w / np.sum(w)
    return x, w

def _nodes_from_params(params):
    """Convert parameter vector to valid nodes in (0,1) strictly increasing."""
    n = len(params)
    # Use tanh to map to (0,1)
    raw = 0.5 * (1.0 + np.tanh(params))
    # Enforce strictly increasing with minimum spacing
    eps = 1e-6
    nodes = np.zeros(n)
    nodes[0] = max(raw[0], eps)
    for i in range(1, n):
        nodes[i] = max(raw[i], nodes[i-1] + eps)
    nodes = np.minimum(nodes, 1.0 - eps)
    # Ensure strictly increasing
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + eps
    return nodes

def _weights_from_params(params):
    """Convert parameter vector to positive weights summing to 1."""
    # Softmax
    log_w = params - np.max(params)
    w = np.exp(log_w)
    w = w / np.sum(w)
    return w

def _objective(params, z_samples, models):
    """Objective: worst-case relative error."""
    n = len(params) // 2
    log_nodes = params[:n]
    log_weights = params[n:]
    
    nodes = _nodes_from_params(log_nodes)
    weights = _weights_from_params(log_weights)
    
    max_err = 0.0
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
    ]
    
    # Z samples for evaluation - focus on regions where error is likely worst
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    
    # Start with Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = _gauss_legendre_nodes(n)
    
    # Convert to parameterization
    # Inverse of tanh: atanh(2*x - 1)
    log_nodes_init = np.arctanh(2.0 * gl_nodes - 1.0)
    log_weights_init = np.log(gl_weights)
    
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Evaluate initial score
    best_params = params_init.copy()
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # L-BFGS-B optimization
    if time.time() < deadline - 15:
        try:
            result = minimize(
                _objective, params_init,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _nodes_from_params(result.x[:n])
                best_weights = _weights_from_params(result.x[n:])
                best_score = result.fun
        except Exception as e:
            pass
    
    # Differential evolution with multiple restarts
    remaining = deadline - time.time()
    if remaining > 40:
        bounds = [(-5, 5)] * n + [(-5, 5)] * n
        
        def de_objective(params):
            return _objective(params, z_samples, models)
        
        # Multiple restarts with different seeds
        n_restarts = 4
        time_per_restart = remaining / n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            try:
                result = differential_evolution(
                    de_objective, bounds,
                    maxiter=80, popsize=20,
                    tol=1e-8, seed=42 + r,
                    polish=True,
                    workers=1,
                    time_limit=max(5, time_per_restart - 3)
                )
                
                if result.fun < best_score:
                    best_params = result.x
                    best_nodes = _nodes_from_params(result.x[:n])
                    best_weights = _weights_from_params(result.x[n:])
                    best_score = result.fun
            except Exception as e:
                pass
    
    # Final refinement with L-BFGS-B
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective, best_params,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _nodes_from_params(result.x[:n])
                best_weights = _weights_from_params(result.x[n:])
                best_score = result.fun
        except Exception as e:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
