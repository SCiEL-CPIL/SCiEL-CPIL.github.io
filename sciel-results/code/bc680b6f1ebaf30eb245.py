"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _E(a, w0, wa):
    """E(a) = H(a)/H0 for CPL dark energy model in flat universe."""
    return (a ** (-3 * (1 + w0 + wa)) + 1e-30) ** (1.0 / 3.0)


def _integrand(x, z, w0, wa):
    """Compute 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E(a, w0, wa)


def _true_integral(z, w0, wa, n_samples=200):
    """Compute reference integral using adaptive Simpson/Gauss-Kronrod."""
    from scipy.integrate import quad
    result, _ = quad(lambda x: _integrand(x, z, w0, wa), 0, 1, limit=100)
    return result


def _worst_case_error(nodes, weights, z_list, w0_list, wa_list):
    """Compute worst relative error across test cosmologies and redshifts."""
    max_err = 0.0
    for z in z_list:
        for w0 in w0_list:
            for wa in wa_list:
                true_val = (1 + z) * z * _true_integral(z, w0, wa)
                quad_val = (1 + z) * z * np.sum(weights * [_integrand(n, z, w0, wa) for n in nodes])
                if abs(true_val) > 1e-15:
                    rel_err = abs(quad_val - true_val) / abs(true_val)
                    max_err = max(max_err, rel_err)
    return max_err


def _objective(params):
    """Objective function for optimization: minimize worst-case error."""
    n = 8
    # First 8 params: nodes (constrained to [0,1] and strictly increasing)
    # Use soft constraints via penalty
    nodes = params[:n]
    weights = params[n:]
    
    # Penalty if nodes not in [0,1]
    if np.any(nodes < 0) or np.any(nodes > 1):
        return 1e10
    
    # Penalty if nodes not strictly increasing
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    
    # Penalty if weights not positive
    if np.any(weights <= 0):
        return 1e10
    
    # Normalize weights
    w_sum = np.sum(weights)
    if abs(w_sum - 1.0) > 1e-8:
        return 1e10 + abs(w_sum - 1.0)
    
    weights_norm = weights / w_sum
    
    # Test on representative cosmologies
    z_list = [0.5, 1.0, 2.0, 5.0]
    w0_list = [-1.0, -0.9, -0.5]
    wa_list = [0.0, 0.3, -0.5]
    
    err = _worst_case_error(nodes, weights_norm, z_list, w0_list, wa_list)
    return err * 100  # Convert to percent


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 8
    
    # Start with Gauss-Legendre as initial guess
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(n)
    nodes_gl = 0.5 * (nodes_gl + 1.0)
    weights_gl = 0.5 * weights_gl
    weights_gl = weights_gl / weights_gl.sum()
    
    # Initial parameters
    x0 = np.concatenate([nodes_gl, weights_gl])
    
    # Bounds for differential evolution
    # Nodes: [0, 1] but need strictly increasing, so use [0.01, 0.99] with small gaps
    bounds = []
    for i in range(n):
        bounds.append((0.001 * (i + 1), 0.999 * (i + 1) / n))
    for i in range(n):
        bounds.append((1e-6, 1.0))
    
    # Run differential evolution with time check
    best_result = None
    best_score = float('inf')
    
    try:
        # Use fewer iterations to stay within time budget
        result = differential_evolution(
            _objective,
            bounds,
            x0=x0,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Not enough time for refinement, use best found or fallback
        if best_result is not None:
            nodes = best_result[:n]
            weights = best_result[n:]
        else:
            nodes = nodes_gl
            weights = weights_gl
    else:
        # Refine with Nelder-Mead if we have time
        if best_result is not None:
            from scipy.optimize import minimize
            try:
                refined = minimize(
                    _objective,
                    best_result,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if refined.fun < best_score:
                    best_score = refined.fun
                    best_result = refined.x
            except Exception:
                pass
        
        nodes = best_result[:n] if best_result is not None else nodes_gl
        weights = best_result[n:] if best_result is not None else weights_gl
    
    # Final cleanup: ensure constraints
    nodes = np.sort(nodes)
    # Ensure strictly increasing with minimum gap
    for i in range(1, n):
        if nodes[i] - nodes[i-1] < 1e-6:
            nodes[i] = nodes[i-1] + 1e-6
    # Clamp to [0,1]
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    
    weights = np.maximum(weights, 1e-10)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
