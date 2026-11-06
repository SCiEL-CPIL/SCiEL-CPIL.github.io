"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _E_of_a(a, w0, w1, Om=0.3, Ode=0.7):
    """Compute E(a) = H(a)/H0 for CPL dark energy model."""
    z = 1.0 / a - 1.0
    # E(z)^2 = Om*(1+z)^3 + Ode*(1+z)^(3*(1+w0+w1*z))
    # For CPL: w(z) = w0 + w1*z
    # E(z) = sqrt(Om*(1+z)^3 + Ode*(1+z)^(3+3*w0+3*w1*z))
    one_plus_z = 1.0 + z
    E2 = Om * one_plus_z**3 + Ode * one_plus_z**(3.0 + 3.0*w0 + 3.0*w1*z)
    return np.sqrt(np.maximum(E2, 1e-30))

def _integrand_at_a(a, w0, w1, Om=0.3, Ode=0.7):
    """1/E(a) as function of a."""
    return 1.0 / _E_of_a(a, w0, w1, Om, Ode)

def _exact_dL(z, w0, w1, Om=0.3, Ode=0.7, n_samples=2000):
    """Compute exact dL/(c/H0) via adaptive-like dense integration."""
    a_vals = 1.0 / (1.0 + z * np.linspace(0, 1, n_samples))
    f_vals = _integrand_at_a(a_vals, w0, w1, Om, Ode)
    # The integral in the formula: ∫_0^1 [1/E(a(t))] dt where a(t) = 1/(1+z*t)
    # Change variable: a = 1/(1+z*t), t = (1/a - 1)/z, dt = -1/(z*a^2) da
    # When t=0, a=1; t=1, a=1/(1+z)
    # ∫_0^1 f(a(t)) dt = ∫_{a=1/(1+z)}^{1} f(a) * (1/(z*a^2)) da
    # Let me just use the t-parameterization directly
    t_vals = np.linspace(0, 1, n_samples)
    a_t = 1.0 / (1.0 + z * t_vals)
    f_t = _integrand_at_a(a_t, w0, w1, Om, Ode)
    integral = np.trapz(f_t, t_vals)
    return (1.0 + z) * z * integral

def _quad_dL(z, nodes, weights, w0, w1, Om=0.3, Ode=0.7):
    """Compute dL/(c/H0) using the quadrature rule."""
    a_vals = 1.0 / (1.0 + z * nodes)
    f_vals = _integrand_at_a(a_vals, w0, w1, Om, Ode)
    integral = np.dot(weights, f_vals)
    return (1.0 + z) * z * integral

def _rel_errors(nodes, weights, test_cosmos):
    """Compute relative errors for a set of test cosmologies."""
    errors = []
    for (w0, w1, Om, Ode) in test_cosmos:
        for z in [0.5, 1.0, 2.0, 3.0, 5.0]:
            exact = _exact_dL(z, w0, w1, Om, Ode)
            quad = _quad_dL(z, nodes, weights, w0, w1, Om, Ode)
            if abs(exact) > 1e-15:
                errors.append(abs(quad - exact) / abs(exact))
    return np.array(errors)

def _objective(params, test_cosmos):
    """Max relative error with penalty for invalid nodes/weights."""
    n = 8
    raw_nodes = params[:n]
    raw_weights = params[n:]
    
    # Transform to ensure nodes in (0,1) strictly increasing and weights positive summing to 1
    # Use cumulative sum of exponentials for nodes
    node_diffs = np.exp(raw_nodes)
    node_cumsum = np.cumsum(node_diffs)
    nodes = node_cumsum / node_cumsum[-1]
    # Ensure strictly in (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Enforce strict increasing
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    # Weights: softmax to ensure positive, then normalize
    w = np.exp(raw_weights - np.max(raw_weights))
    weights = w / w.sum()
    
    errors = _rel_errors(nodes, weights, test_cosmos)
    return np.max(errors)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 8
    
    # Define test cosmologies (CPL models)
    test_cosmos = [
        (0.0, 0.0, 0.3, 0.7),      # LambdaCDM
        (-1.0, 0.0, 0.3, 0.7),     # Cosmological constant
        (-1.0, 0.5, 0.3, 0.7),     # CPL w0=-1, w1=0.5
        (-1.0, -0.5, 0.3, 0.7),    # CPL w0=-1, w1=-0.5
        (-0.5, 0.0, 0.3, 0.7),     # w0=-0.5
        (-1.5, 0.0, 0.3, 0.7),     # w0=-1.5
        (-1.0, 0.3, 0.25, 0.75),   # Different Om
        (-1.0, 0.3, 0.35, 0.65),   # Different Om
        (-0.8, 0.2, 0.3, 0.7),     # General CPL
        (-1.2, -0.3, 0.3, 0.7),    # General CPL
        (-1.0, 1.0, 0.3, 0.7),     # Strong w1
        (-1.0, -1.0, 0.3, 0.7),    # Strong negative w1
    ]
    
    # Initial guess: Chebyshev-Gauss quadrature mapped to [0,1]
    k = np.arange(1, n + 1)
    x_cheb = np.cos((2*k - 1) * np.pi / (2*n))  # in [-1, 1]
    nodes0 = (x_cheb + 1.0) / 2.0  # in [0, 1]
    nodes0 = np.sort(nodes0)
    weights0 = np.full(n, 1.0/n)
    
    # Inverse transform for initial params
    # For nodes: node_cumsum / total = nodes0, so node_diffs[i] proportional to nodes0[i] - nodes0[i-1]
    node_diffs0 = np.diff(np.concatenate([[0], nodes0]))
    raw_nodes0 = np.log(np.maximum(node_diffs0, 1e-10))
    raw_weights0 = np.log(weights0)
    
    params0 = np.concatenate([raw_nodes0, raw_weights0])
    
    best_params = params0.copy()
    best_score = _objective(params0, test_cosmos)
    
    # Try a few optimization methods
    methods_to_try = ['Nelder-Mead', 'Powell']
    
    for method in methods_to_try:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _objective, params0, args=(test_cosmos,),
                method=method,
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
            params0 = result.x.copy()  # Use result as next starting point
        except Exception:
            pass
    
    # Extract best nodes and weights
    raw_nodes = best_params[:n]
    raw_weights = best_params[n:]
    
    node_diffs = np.exp(raw_nodes)
    node_cumsum = np.cumsum(node_diffs)
    nodes = node_cumsum / node_cumsum[-1]
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    w = np.exp(raw_weights - np.max(raw_weights))
    weights = w / w.sum()
    
    # Final validation
    if time.time() < deadline:
        # One more refinement with tighter tolerance
        try:
            result = minimize(
                _objective, best_params, args=(test_cosmos,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                raw_nodes = best_params[:n]
                raw_weights = best_params[n:]
                node_diffs = np.exp(raw_nodes)
                node_cumsum = np.cumsum(node_diffs)
                nodes = node_cumsum / node_cumsum[-1]
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                for i in range(1, n):
                    if nodes[i] <= nodes[i-1]:
                        nodes[i] = nodes[i-1] + 1e-10
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                w = np.exp(raw_weights - np.max(raw_weights))
                weights = w / w.sum()
        except Exception:
            pass
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
