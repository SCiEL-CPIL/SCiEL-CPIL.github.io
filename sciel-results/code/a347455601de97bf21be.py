"""Optimized 8-node positive quadrature for luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _E_of_z(z, w0, wa):
    """E(a) = H(a)/H0 for flat CPL model with matter + dark energy."""
    a = 1.0 / (1.0 + z)
    # H^2/H0^2 = Omega_m a^{-3} + Omega_de * exp(3(1+w0+wa)(a-1))
    # For flat: Omega_de = 1 - Omega_m
    # But we need to handle the case where Omega_m might be different.
    # Standard CPL: w(a) = w0 + wa*(1-a)
    # rho_de(a) = rho_de0 * exp(3 * integral_1^a (1+w(a'))/a' da')
    # integral = (1+w0)*ln(a) + wa*(a-1)
    # So rho_de(a)/rho_de0 = exp(3*((1+w0)*ln(a) + wa*(a-1)))
    # = a^(3*(1+w0)) * exp(3*wa*(a-1))
    # E^2(a) = Omega_m * a^{-3} + (1-Omega_m) * a^(3*(1+w0)) * exp(3*wa*(a-1))
    om_m = 0.3  # Typical value
    om_de = 1.0 - om_m
    term_m = om_m * a**(-3)
    term_de = om_de * (a**(3.0 * (1.0 + w0))) * np.exp(3.0 * wa * (a - 1.0))
    E2 = term_m + term_de
    return np.sqrt(E2)


def _dL_norm(z, w0, wa):
    """Normalized luminosity distance dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')"""
    # Use adaptive quadrature as reference
    from scipy.integrate import quad
    f = lambda zp: 1.0 / _E_of_z(zp, w0, wa)
    integral, _ = quad(f, 0, z, limit=100)
    return (1.0 + z) * integral


def _quad_error(nodes, weights, w0, wa, z_max=2.0, n_samples=50):
    """Compute worst relative error over a grid of z values."""
    z_vals = np.linspace(0.1, z_max, n_samples)
    max_err = 0.0
    for z in z_vals:
        # True value
        true_val = _dL_norm(z, w0, wa)
        if true_val == 0:
            continue
        # Quadrature approximation
        # dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
        # With substitution: integral = sum_i weights_i * (1/E(a_i)) * (dz/da conversion)
        # Actually the formula given is: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # Wait, let me re-read: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
        # where a_i = 1/(1+z*nodes[i])
        # This is a specific quadrature form. Let's use it as given.
        a_i = 1.0 / (1.0 + z * nodes)
        E_i = _E_of_z(z * nodes, w0, wa)  # E at z_i = z * nodes[i]
        approx = (1.0 + z) * z * np.sum(weights / E_i)
        rel_err = abs(approx - true_val) / true_val
        max_err = max(max_err, rel_err)
    return max_err


def _objective(params):
    """Objective: worst-case relative error across several CPL models."""
    nodes = params[:8]
    weights = params[8:16]
    
    # Ensure nodes are in [0,1] and increasing
    # Transform from unconstrained to constrained
    # For now, assume params are already constrained
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e6
    if np.any(np.diff(nodes) <= 0):
        return 1e6
    if np.any(weights <= 0):
        return 1e6
    if abs(np.sum(weights) - 1.0) > 1e-10:
        return 1e6
    
    # Test on several CPL models
    test_models = [
        (0.0, 0.0),    # LCDM
        (-0.5, 0.0),   # w0=-0.5
        (0.5, 0.0),    # w0=0.5
        (-0.5, 0.5),   # w0=-0.5, wa=0.5
        (-0.5, -0.5),  # w0=-0.5, wa=-0.5
        (0.0, 1.0),    # w0=0, wa=1
        (0.0, -1.0),   # w0=0, wa=-1
    ]
    
    max_err = 0.0
    for w0, wa in test_models:
        err = _quad_error(nodes, weights, w0, wa, z_max=1.5, n_samples=20)
        max_err = max(max_err, err)
    
    return max_err


def _transform_params(x):
    """Transform unconstrained params to constrained nodes and weights."""
    # Nodes: use sigmoid to map to (0,1), then sort
    raw_nodes = np.tanh(x[:8])
    nodes = 0.5 * (raw_nodes + 1.0)  # Map to (0,1)
    nodes = np.sort(nodes)
    # Ensure strictly increasing by adding small epsilon
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-6:
            nodes[i] = nodes[i-1] + 1e-6
    
    # Weights: use softmax to ensure positive and sum to 1
    raw_weights = x[8:16]
    weights = np.exp(raw_weights - np.max(raw_weights))
    weights = weights / np.sum(weights)
    
    return nodes, weights


def _objective_transformed(x):
    """Objective with parameter transformation."""
    nodes, weights = _transform_params(x)
    return _objective(np.concatenate([nodes, weights]))


def construct():
    start_time = time.time()
    deadline = 75.0  # 75s deadline, 15s safety margin
    
    # Start with composite midpoint rule
    n = 8
    nodes_init = (np.arange(n) + 0.5) / n
    weights_init = np.full(n, 1.0 / n)
    
    best_params = np.concatenate([nodes_init, weights_init])
    best_err = _objective(best_params)
    
    # Try differential evolution for a short time
    if time.time() - start_time < deadline - 30:
        try:
            # Initialize in unconstrained space
            x0 = np.zeros(16)
            # Map initial nodes and weights to unconstrained space
            raw_nodes = 2.0 * nodes_init - 1.0
            x0[:8] = np.arctanh(np.clip(raw_nodes, -0.999, 0.999))
            x0[8:16] = np.log(weights_init)
            
            result = differential_evolution(
                _objective_transformed,
                bounds=[(-5, 5)] * 16,
                seed=42,
                maxiter=50,
                tol=1e-6,
                polish=False,
                init='latinhypercube'
            )
            if result.fun < best_err:
                best_err = result.fun
                best_params = np.concatenate(_transform_params(result.x))
        except Exception:
            pass
    
    # Try local refinement with Nelder-Mead
    if time.time() - start_time < deadline - 10:
        try:
            x_best = np.zeros(16)
            raw_nodes = 2.0 * best_params[:8] - 1.0
            x_best[:8] = np.arctanh(np.clip(raw_nodes, -0.999, 0.999))
            x_best[8:16] = np.log(best_params[8:16])
            
            result = minimize(
                _objective_transformed,
                x_best,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_err:
                best_err = result.fun
                best_params = np.concatenate(_transform_params(result.x))
        except Exception:
            pass
    
    # Final check
    nodes, weights = best_params[:8], best_params[8:16]
    
    # Ensure constraints are met
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-8:
            nodes[i] = nodes[i-1] + 1e-8
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
