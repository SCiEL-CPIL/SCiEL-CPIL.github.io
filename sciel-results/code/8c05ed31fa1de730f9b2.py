"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

# Deadline guard: kill after 80s to stay safely under 90s limit
_DEADLINE = time.time() + 80.0


def _time_left():
    return _DEADLINE - time.time()


def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat LambdaCDM/CPL model.
    w(a) = w0 + wa*(1-a)
    E^2 = Om_m * a^-3 + (1-Om_m) * exp(-3*integral of (1+w) da'/a' from 1 to a)
    For CPL: integral of (1+w(a')) da'/a' = (1+w0) ln a + wa * (a-1)/a
    So E^2 = Om_m/a^3 + (1-Om_m) * a^(-3*(1+w0)) * exp(-3*wa*(a-1)/a)
    """
    a = np.asarray(a, dtype=float)
    # Handle a=0 carefully; in practice a is in (0,1]
    # E^2 = Om_m * a^-3 + (1-Om_m) * a^(-3*(1+w0)) * exp(-3*wa*(a-1)/a)
    # Use log form for numerical stability
    log_term1 = np.log(0.3) - 3.0 * np.log(a)
    log_term2 = np.log(0.7) - 3.0 * (1.0 + w0) * np.log(a) - 3.0 * wa * (a - 1.0) / a
    # Use log-sum-exp for stability
    m = np.maximum(log_term1, log_term2)
    E2 = np.exp(m) * (np.exp(log_term1 - m) + np.exp(log_term2 - m))
    return np.sqrt(E2)


def _integrand_z(z, w0, wa):
    """Integrand for dL/(c/H0) = (1+z) * z * integral_0^z dz'/E(z')
    Actually the formula is: dL/(c/H0) = (1+z) * integral_0^z dz' / E(z')
    Wait, let me re-read the task.
    
    The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
    
    Hmm, this looks like the quadrature is applied to the integral part.
    Let me think about what the integral is.
    
    Standard luminosity distance: dL = (c/H0) * (1+z) * integral_0^z dz'/E(z')
    
    So the quadrature approximates integral_0^z dz'/E(z') = sum_i weights[i] / E(a_i) * (z * nodes[i] mapping?)
    
    Wait, the nodes are in [0,1] and a_i = 1/(1+z*nodes[i]). So the variable of integration is mapped from [0,1] to z' in [0,z].
    
    If we let t be in [0,1], then z' = z*t, so dz' = z*dt, and a' = 1/(1+z*t).
    
    integral_0^z dz'/E(z') = integral_0^1 z*dt / E(1/(1+z*t)) = z * integral_0^1 dt / E(a(t))
    
    So dL/(c/H0) = (1+z) * z * integral_0^1 dt / E(a(t))
    
    And the quadrature approximates integral_0^1 dt / E(a(t)) ≈ sum_i weights[i] / E(a_i)
    
    So the full expression is: dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)
    
    That matches the task description!
    """
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_of_a(a, w0, wa)


def _reference_dL(z, w0, wa):
    """Reference luminosity distance using adaptive quadrature."""
    # integral_0^z dz'/E(z') = z * integral_0^1 dt / E(1/(1+z*t))
    def inner(t):
        a = 1.0 / (1.0 + z * t)
        return 1.0 / _E_of_a(a, w0, wa)
    
    val, _ = quad(inner, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val


def _quadrature_dL(z, nodes, weights, w0, wa):
    """Quadrature-based luminosity distance."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)


def construct():
    t_start = time.time()
    
    # Pre-compute reference values for a set of test cosmologies and redshifts
    # Use a diverse set of CPL parameters
    test_cosmos = [
        (0.0, 0.0),    # LambdaCDM
        (-0.5, 0.0),   # Quintessence-like
        (-1.0, 1.0),   # CPL with evolution
        (-0.8, 0.5),   # Another CPL
        (-0.3, -0.5),  # Phantom-ish
        (-1.5, 2.0),   # Strong evolution
    ]
    
    test_zs = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    
    # Pre-compute reference integrals
    ref_data = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            if _time_left() < 15:
                break
            ref_val = _reference_dL(z, w0, wa)
            ref_data.append((z, w0, wa, ref_val))
    
    if len(ref_data) == 0:
        # Fallback: return midpoint rule
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Objective: maximize worst relative error (minimize negative of it)
    # We want to minimize the worst relative error in percent
    
    def _objective(params):
        """params: 16 values = 8 nodes (logit-transformed for positivity) + 8 weights (log-transformed)
        Nodes are in logit space: node = sigmoid(logit_param)
        Weights are in log space: weight = exp(log_weight_param)
        """
        logit_nodes = params[:8]
        log_weights = params[8:16]
        
        # Convert to actual nodes and weights
        nodes = 1.0 / (1.0 + np.exp(-logit_nodes))  # sigmoid
        # Ensure strictly increasing and in (0,1)
        # The sigmoid maps R -> (0,1), and if logit_nodes are increasing, nodes are increasing
        # But we need to ensure they're strictly increasing. Let's sort and check.
        
        # Actually, let's use a different parametrization to ensure ordering.
        # Use cumulative sum of positive values.
        # params[:8] are positive (via softplus or exp), nodes are cumulative sums normalized.
        
        # Let me redo: use 7 positive increments + 1 for the last
        # Actually simpler: use logit for each node independently but enforce ordering via penalty or reparam.
        
        # Better approach: parametrize nodes as expit of sorted parameters
        # Or use: node_i = sigmoid(cumsum of positive params)
        
        # Let's use: raw_nodes = params[:8], transform via sigmoid, then sort
        # But sorting breaks differentiability. Instead, use the fact that we can just
        # work with unconstrained and add penalty for ordering violation.
        
        # Simplest robust approach: use the "cumulative positive" trick
        # Let p_i = exp(params[i]) for i=0..7, all positive
        # nodes = cumsum(p) / sum(p), which gives 8 values in (0,1) strictly increasing
        # But this uses 8 params for 8 nodes, and the last node is always 1.0, which is not in (0,1).
        
        # Alternative: use 7 params for 7 gaps, nodes[0] = sigmoid(params[0]), 
        # nodes[i] = nodes[i-1] + sigmoid(params[i])*(1-nodes[i-1])
        # This ensures 0 < nodes[0] < nodes[1] < ... < nodes[7] < 1
        
        # Let me use this approach:
        # params[0:8] control the nodes via sequential construction
        # params[8:16] control weights via exp
        
        # Rebuild nodes from params[0:8]
        n_params = params[:8]
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-n_params[0]))  # in (0,1)
        for i in range(1, 8):
            # Fraction of remaining interval
            frac = 1.0 / (1.0 + np.exp(-n_params[i]))  # in (0,1)
            nodes[i] = nodes[i-1] + frac * (1.0 - nodes[i-1])
        
        # Nodes are now strictly in (0,1) and strictly increasing
        
        # Weights: positive via exp, then normalize
        raw_weights = np.exp(params[8:16])
        weights = raw_weights / np.sum(raw_weights)
        
        # Compute worst relative error
        max_rel_err = 0.0
        for z, w0, wa, ref_val in ref_data:
            if ref_val == 0:
                continue
            quad_val = _quadrature_dL(z, nodes, weights, w0, wa)
            rel_err = abs(quad_val - ref_val) / ref_val
            if rel_err > max_rel_err:
                max_rel_err = rel_err
        
        return max_rel_err * 100.0  # in percent
    
    # Initial guess: midpoint rule
    n = 8
    init_nodes = (np.arange(n) + 0.5) / n
    # Convert to our parametrization
    init_params_nodes = np.zeros(8)
    init_params_nodes[0] = np.log(init_nodes[0] / (1 - init_nodes[0]))  # logit
    for i in range(1, 8):
        frac = (init_nodes[i] - init_nodes[i-1]) / (1 - init_nodes[i-1])
        init_params_nodes[i] = np.log(frac / (1 - frac))
    
    init_weights = np.full(8, 1.0 / 8)
    init_params_weights = np.log(init_weights)
    
    x0 = np.concatenate([init_params_nodes, init_params_weights])
    
    best_params = x0.copy()
    best_obj = _objective(x0)
    
    # Try multiple optimization methods
    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']
    
    for method in methods:
        if _time_left() < 10:
            break
        
        try:
            if method == 'L-BFGS-B':
                result = minimize(_objective, x0, method=method, options={'maxiter': 500, 'ftol': 1e-12})
            elif method == 'Powell':
                result = minimize(_objective, x0, method=method, options={'maxiter': 500, 'ftol': 1e-12})
            else:
                result = minimize(_objective, x0, method=method, options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract best nodes and weights
    n_params = best_params[:8]
    nodes = np.zeros(8)
    nodes[0] = 1.0 / (1.0 + np.exp(-n_params[0]))
    for i in range(1, 8):
        frac = 1.0 / (1.0 + np.exp(-n_params[i]))
        nodes[i] = nodes[i-1] + frac * (1.0 - nodes[i-1])
    
    raw_weights = np.exp(best_params[8:16])
    weights = raw_weights / np.sum(raw_weights)
    
    # Ensure nodes are strictly in (0,1) and strictly increasing
    # Clip to safe range
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    
    # Normalize weights to sum to exactly 1
    weights = weights / np.sum(weights)
    
    # Verify weights are positive
    if np.any(weights <= 0):
        # Fallback to midpoint
        return {"nodes": (np.arange(8) + 0.5) / 8, "weights": np.full(8, 1.0 / 8)}
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
