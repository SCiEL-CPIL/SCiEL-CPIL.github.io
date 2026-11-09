"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Uses moment-matching against a representative set of CPL (Chevallier-Polarski-Linder)
    dark energy models to find nodes and weights that minimize worst-case relative error.
    """
    t_start = time.time()
    deadline = t_start + 80.0  # Safety margin before 90s kill
    
    # Representative CPL models: (w0, wa) pairs
    # These span a range of dark energy behaviors
    cpl_models = [
        (0.0, 0.0),    # Lambda CDM
        (-0.5, 0.5),   # Quintessence-like
        (-1.5, 1.0),   # Phantom-like
        (-1.0, 0.5),   # Intermediate
        (-0.8, 0.2),   # Mild deviation
        (-1.2, -0.3),  # Another variant
        (-0.3, -0.7),  # Another
        (-1.0, -1.0),  # Another
    ]
    
    # Redshifts to test (disjoint from training, but representative)
    z_test = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    def E_of_a(a, w0, wa):
        """H(a)/H0 for flat CPL model: E^2 = a^(-6) + a^(-3(1+w0+wa)) * a^(-3wa)"""
        # For CPL: E^2(a) = a^(-6) + a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        # Wait, let me be more careful. The CPL parametrization is:
        # w(a) = w0 + wa*(1-a)
        # E^2(a) = a^(-6) + a^(-3(1+w0+wa)) * a^(-3wa) ... 
        # Actually: E^2(a) = a^(-6) + Omega_m * a^(-3) + Omega_DE * a^(-3(1+w(a)))
        # For flat: Omega_m + Omega_DE = 1, let's use Omega_m = 0.3, Omega_DE = 0.7
        # E^2(a) = 0.3 * a^(-3) + 0.7 * a^(-3*(1+w0+wa*(1-a)))
        # But the task says the evaluator computes integrands itself, so I just need
        # to match the form. Let me use a standard flat CPL with Omega_m=0.3.
        Om = 0.3
        Ode = 0.7
        w = w0 + wa * (1 - a)
        return np.sqrt(Om * a**(-3) + Ode * a**(-3 * (1 + w)))
    
    def luminosity_distance_ratio(z, nodes, weights, w0, wa):
        """Compute dL/(c/H0) = (1+z) * integral_0^z dz'/E(1/(1+z')) using quadrature."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        integral = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral
    
    def exact_luminosity_distance(z, w0, wa):
        """Compute exact luminosity distance via adaptive quadrature."""
        from scipy.integrate import quad
        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return 1.0 / E_of_a(a, w0, wa)
        val, _ = quad(integrand, 0, z, limit=200)
        return (1.0 + z) * z * val
    
    def worst_case_error(nodes, weights):
        """Compute worst-case relative error across all test models and redshifts."""
        max_err = 0.0
        for (w0, wa) in cpl_models:
            for z in z_test:
                exact = exact_luminosity_distance(z, w0, wa)
                approx = luminosity_distance_ratio(z, nodes, weights, w0, wa)
                if exact > 0:
                    rel_err = abs(approx - exact) / exact
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err * 100.0  # in percent
    
    # Initial guess: Gauss-Legendre on [0,1]
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(8)
    nodes_init = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    
    # Optimize using scipy minimize
    # Parameters: 8 nodes + 8 weights = 16 params
    # Constraints: nodes in (0,1) strictly increasing, weights > 0, sum(weights) = 1
    
    def objective(params):
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints - return large penalty if violated
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e6
        
        return worst_case_error(nodes, weights)
    
    # Use Nelder-Mead which doesn't require gradients and handles constraints via penalty
    best_params = None
    best_obj = np.inf
    
    # Start from Gauss-Legendre
    params_init = np.concatenate([nodes_init, weights_init])
    
    try:
        result = minimize(
            objective,
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time, return best found or initial
        if best_params is not None:
            nodes = best_params[:8]
            weights = best_params[8:]
        else:
            nodes = nodes_init
            weights = weights_init
        # Normalize weights
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}
    
    # Try a few more restarts if time permits
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        # Perturb the initial guess
        nodes_pert = nodes_init + np.random.randn(8) * 0.05
        nodes_pert = np.clip(nodes_pert, 0.01, 0.99)
        nodes_pert = np.sort(nodes_pert)
        weights_pert = weights_init + np.random.randn(8) * 0.05
        weights_pert = np.abs(weights_pert)
        weights_pert = weights_pert / np.sum(weights_pert)
        
        params_pert = np.concatenate([nodes_pert, weights_pert])
        
        try:
            result = minimize(
                objective,
                params_pert,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final result
    if best_params is not None:
        nodes = best_params[:8]
        weights = best_params[8:]
    else:
        nodes = nodes_init
        weights = weights_init
    
    # Ensure constraints are satisfied
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    
    weights = np.abs(weights)
    if np.sum(weights) < 1e-12:
        weights = np.full(8, 1.0/8)
    else:
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
