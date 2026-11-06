"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time

def _E_of_a(a, w0, wa, Om0=0.3):
    """E(a) = H(a)/H0 for CPL model in flat universe."""
    # H(a)/H0 = sqrt(Om0 * a^{-3} + (1-Om0) * a^{-3(1+w0+wa)} * a^{wa})
    # More precisely: E(a)^2 = Om0 * a^{-3} + (1-Om0) * (a^{-3} * a^{wa})^{-1} ... 
    # CPL: w(z) = w0 + wa*z, z = 1/a - 1
    # E(a)^2 = Om0 * a^{-3} + (1-Om0) * exp(3*(1+w0+wa)*(1/a - 1)) * a^{-3*(1+w0+wa)}
    # Simpler form: E(a)^2 = Om0/a^3 + (1-Om0)/a^3 * a^{wa} * exp(-3*wa*(1/a-1))
    # Standard CPL: E(a)^2 = Om0 * a^{-3} + (1-Om0) * a^{-3-3*wa} * exp(3*wa*(1/a - 1))
    # Let me use: E(a)^2 = Om0 * a^{-3} + (1-Om0) * (1/a)^{3*(1+w0+wa)} * exp(3*wa*(1/a - 1))
    z = 1.0/a - 1.0
    w = w0 + wa * z
    # E(a)^2 = Om0 * a^{-3} + (1-Om0) * a^{-3(1+w)} * exp(3*w*a*(1/a - 1))... 
    # Standard: E(a)^2 = Om0/a^3 + (1-Om0) * exp(3*(1+w0+wa)*(1/a-1)) / a^{3*(1+w0+wa)}
    # Let me use the standard CPL form:
    # E(a)^2 = Om0 * a^{-3} + (1-Om0) * a^{-3(1+w0+wa)} * exp(3*wa*(1/a - 1))
    term1 = Om0 * a**(-3)
    term2 = (1 - Om0) * a**(-3*(1 + w0 + wa)) * np.exp(3 * wa * (1.0/a - 1.0))
    return np.sqrt(term1 + term2)

def _exact_integral(z_target, w0, wa, Om0=0.3):
    """Compute exact ∫₀^z dz'/E(z') using adaptive quadrature."""
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return 1.0 / _E_of_a(a, w0, wa, Om0)
    result, _ = quad(integrand, 0, z_target, limit=200)
    return result

def _quadrature_integral(z_target, nodes, weights, w0, wa, Om0=0.3):
    """Compute quadrature approximation of ∫₀^z dz'/E(z')."""
    # z' = z_target * t, so a = 1/(1 + z_target * t)
    z_vals = z_target * nodes
    a_vals = 1.0 / (1.0 + z_vals)
    E_vals = _E_of_a(a_vals, w0, wa, Om0)
    return z_target * np.sum(weights / E_vals)

def _relative_errors(nodes, weights, models, z_values, Om0=0.3):
    """Compute relative errors for given nodes/weights across models and z."""
    errors = []
    for w0, wa in models:
        for z in z_values:
            exact = _exact_integral(z, w0, wa, Om0)
            approx = _quadrature_integral(z, nodes, weights, w0, wa, Om0)
            if abs(exact) > 1e-12:
                rel_err = abs(approx - exact) / abs(exact)
                errors.append(rel_err)
    return np.array(errors)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Diverse set of CPL models (w0, wa)
    models = [
        (0.0, 0.0),      # LambdaCDM
        (-1.0, 0.0),     # Cosmological constant
        (-0.5, -0.5),    # Quintessence-like
        (-1.0, 0.5),     # Phantom crossing
        (-1.5, 1.0),     # Strong phantom
        (-0.8, -0.3),    # Mild quintessence
        (-1.2, 0.8),     # Phantom
        (-0.9, -0.5),    # Another quintessence
        (-1.0, -1.0),    # Exotic
        (-0.7, 0.2),     # Mild
    ]
    
    # Redshift values to test
    z_values = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    Om0 = 0.3
    
    # Initial guess: Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (x_gl + 1.0)  # Map from [-1,1] to [0,1]
    weights_init = w_gl / 2.0
    
    # Ensure nodes are strictly increasing and in (0,1)
    nodes_init = np.sort(nodes_init)
    # Clip to avoid 0 and 1
    nodes_init = np.clip(nodes_init, 1e-6, 1 - 1e-6)
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        # params: 8 nodes (in [0,1]) + 8 weights (positive)
        nodes = params[:8]
        log_weights = params[8:]
        
        # Ensure nodes are in (0,1) and increasing
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        
        weights = np.exp(log_weights)
        # Normalize weights to sum to 1
        w_sum = np.sum(weights)
        if w_sum <= 0:
            return 1e6
        weights = weights / w_sum
        
        errors = _relative_errors(nodes, weights, models, z_values, Om0)
        return np.max(errors)
    
    # Use differential evolution for global optimization
    # Bounds: nodes in [1e-4, 1-1e-4], log_weights in [-10, 10]
    bounds = []
    for i in range(8):
        bounds.append((1e-4, 1 - 1e-4))
    for i in range(8):
        bounds.append((-10, 10))
    
    x0 = np.concatenate([nodes_init, np.log(weights_init)])
    
    try:
        result = differential_evolution(
            objective, bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            polish=True,
            workers=-1,
            init='sobol'
        )
    except Exception:
        result = None
    
    if result is not None and result.success:
        nodes_opt = result.x[:8]
        log_weights_opt = result.x[8:]
        weights_opt = np.exp(log_weights_opt)
        weights_opt = weights_opt / np.sum(weights_opt)
    else:
        nodes_opt = nodes_init
        weights_opt = weights_init
    
    # Final refinement with local optimization
    def local_objective(params):
        if time.time() > deadline:
            return 1e6
        nodes = params[:8]
        log_weights = params[8:]
        
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        
        weights = np.exp(log_weights)
        w_sum = np.sum(weights)
        if w_sum <= 0:
            return 1e6
        weights = weights / w_sum
        
        errors = _relative_errors(nodes, weights, models, z_values, Om0)
        return np.max(errors)
    
    x0_local = np.concatenate([nodes_opt, np.log(weights_opt)])
    try:
        local_result = minimize(
            local_objective, x0_local,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if local_result.success and local_result.fun < 1e-3:
            nodes_opt = local_result.x[:8]
            weights_opt = np.exp(local_result.x[8:])
            weights_opt = weights_opt / np.sum(weights_opt)
    except Exception:
        pass
    
    # Final validation: ensure nodes strictly increasing in (0,1), weights positive, sum to 1
    nodes_final = np.sort(nodes_opt)
    nodes_final = np.clip(nodes_final, 1e-8, 1 - 1e-8)
    # Ensure strict increase
    for i in range(1, 8):
        if nodes_final[i] <= nodes_final[i-1]:
            nodes_final[i] = nodes_final[i-1] + 1e-6
    
    weights_final = np.abs(weights_opt)
    weights_final = weights_final / np.sum(weights_final)
    
    # Ensure sum is exactly 1 within 1e-12
    s = np.sum(weights_final)
    weights_final = weights_final / s
    
    return {"nodes": nodes_final, "weights": weights_final}
# EVOLVE-BLOCK-END
