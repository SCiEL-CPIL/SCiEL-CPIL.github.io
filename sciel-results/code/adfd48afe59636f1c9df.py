import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def _E(a, w0, w1):
    """CPL E(a) = H(a)/H0"""
    # E(a) = a^{(3/2)(w0+w1-1)} * exp(3w1/2 * (1-a))
    p = 1.5 * (w0 + w1 - 1.0)
    q = 1.5 * w1
    return np.power(a, p) * np.exp(q * (1.0 - a))

def _true_integral(z, w0, w1):
    """Compute true dL/(c/H0) = (1+z)*z*∫_0^1 dt/E(a(t))"""
    def integrand(t):
        a = 1.0 / (1.0 + z * t)
        return 1.0 / _E(a, w0, w1)
    val, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val

def _quad_error(nodes, weights, z, w0, w1):
    """Compute quadrature estimate and relative error"""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E(a, w0, w1)
    quad_val = np.dot(weights, 1.0 / E_vals)
    true_val = (1.0 + z) * z * quad_val  # This is the quadrature estimate
    # True value:
    def integrand(t):
        a_t = 1.0 / (1.0 + z * t)
        return 1.0 / _E(a_t, w0, w1)
    true_int, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    true_val = (1.0 + z) * z * true_int
    return abs(true_val - quad_val) / true_val

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Generate diverse test cases
    w0_vals = np.array([-1.5, -1.2, -1.0, -0.8, -0.5])
    w1_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    z_vals = np.logspace(np.log10(0.1), np.log10(3.0), 15)
    
    test_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                test_cases.append((z, w0, w1))
    
    # Pre-compute true values
    true_vals = []
    for z, w0, w1 in test_cases:
        tv = _true_integral(z, w0, w1)
        true_vals.append(tv)
    true_vals = np.array(true_vals)
    
    n_tests = len(test_cases)
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    w1_arr = np.array([tc[2] for tc in test_cases])
    
    def _eval_all_errors(nodes, weights):
        """Vectorized evaluation of relative errors for all test cases"""
        errors = np.zeros(n_tests)
        for i in range(n_tests):
            z = z_arr[i]
            a = 1.0 / (1.0 + z * nodes)
            p = 1.5 * (w0_arr[i] + w1_arr[i] - 1.0)
            q = 1.5 * w1_arr[i]
            E_vals = np.power(a, p) * np.exp(q * (1.0 - a))
            quad_val = np.dot(weights, 1.0 / E_vals)
            quad_full = (1.0 + z) * z * quad_val
            errors[i] = abs(true_vals[i] - quad_full) / true_vals[i]
        return errors
    
    def _objective(params):
        x = params[:8]
        y = params[8:]
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.sort(nodes)
        # Check nodes are in (0,1) and strictly increasing
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        weights = np.exp(y - np.max(y))
        weights /= np.sum(weights)
        errors = _eval_all_errors(nodes, weights)
        return np.max(errors)
    
    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = 0.5 * (gl_nodes + 1.0)  # Map from [-1,1] to [0,1]
    gl_weights = 0.5 * gl_weights
    
    # Convert to parameterization
    x0 = np.log(gl_nodes / (1.0 - gl_nodes))  # Inverse sigmoid
    y0 = np.log(gl_weights)  # Inverse softmax (approximately)
    
    params0 = np.concatenate([x0, y0])
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # Optimize
    result = minimize(_objective, params0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result.fun < best_score:
        best_score = result.fun
        best_params = result.x
    
    # Check time
    if time.time() < deadline:
        # Try Powell as well
        result2 = minimize(_objective, best_params, method='Powell',
                          options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-12})
        if result2.fun < best_score:
            best_score = result2.fun
            best_params = result2.x
    
    # Extract best nodes and weights
    x_best = best_params[:8]
    y_best = best_params[8:]
    nodes = 1.0 / (1.0 + np.exp(-x_best))
    nodes = np.sort(nodes)
    weights = np.exp(y_best - np.max(y_best))
    weights /= np.sum(weights)
    
    # Ensure nodes are strictly in (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    nodes = np.sort(nodes)
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
