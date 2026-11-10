"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Structural Change: 
    Instead of fragile global non-convex optimization of nodes and weights, 
    we use Gauss-Legendre nodes (optimal for smooth functions) and perform a 
    safe, local refinement of the weights only, guarded against numerical issues.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # 1. Initialize with Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = roots_legendre(8)
    # Map from [-1, 1] to [0, 1]
    nodes = 0.5 * (gl_nodes + 1.0)
    weights = 0.5 * gl_weights
    
    # Ensure strict increasing and within (0,1)
    # GL nodes are already strictly increasing in (-1, 1), so mapped to (0, 1)
    # Weights are positive and sum to 1
    
    # 2. Define representative cosmologies for weight refinement
    # We only optimize weights, keeping nodes fixed. This is a convex-ish problem 
    # if we use L2 error, or we can use a robust local search for max error.
    
    Om_m = 0.3
    Om_de = 0.7
    w0_values = [-1.0, -0.8, -0.5, 0.0]
    wa_values = [-0.5, 0.0, 0.5]
    z_test = [0.5, 1.0, 2.0]
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=10001):
        """High-accuracy reference integral."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-1:2]))
        return z * integral
    
    # Pre-compute reference values
    ref_vals = []
    for w0 in w0_values:
        for wa in wa_values:
            for z in z_test:
                try:
                    rv = reference_integral(z, w0, wa)
                    if rv > 1e-12: # Guard against zero reference
                        ref_vals.append(rv)
                except Exception:
                    pass
    
    if not ref_vals:
        return {"nodes": nodes, "weights": weights}
        
    ref_vals = np.array(ref_vals)
    
    # Objective: Minimize worst-case relative error
    # We optimize weights only. Nodes are fixed.
    # Parameterization: w_i = exp(y_i) / sum(exp(y_j))
    
    def objective(y):
        if time.time() > deadline:
            return 1e10
        w_exp = np.exp(y - np.max(y))
        w_sum = np.sum(w_exp)
        if w_sum < 1e-10:
            return 1e10
        w = w_exp / w_sum
        
        # Compute quadrature values for all reference cases
        # We need to re-compute E_vals for the fixed nodes for each case
        # This is inefficient, so we pre-compute the 1/E values at the nodes
        
        # Pre-computation outside objective is better, but we need to handle 
        # different cosmologies. Let's just calculate the error for a subset 
        # to keep it fast, or vectorize.
        
        max_err = 0.0
        # Iterate over a small subset of representative cases to keep objective fast
        # We'll pick 3 cases: (w0=-1, wa=0, z=1), (w0=0, wa=0, z=1), (w0=-1, wa=0.5, z=2)
        cases = [
            (-1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.5, 2.0)
        ]
        
        for w0, wa, z in cases:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = E_of_a(a_vals, w0, wa)
            quad_val = np.sum(w / E_vals) * z
            
            # Find corresponding reference
            # We didn't store indices, so we just re-calc ref for these 3 specific cases
            # This is safe and fast enough
            ref_val = reference_integral(z, w0, wa, n_ref=5001)
            if ref_val > 1e-12:
                err = abs(quad_val - ref_val) / ref_val
                if err > max_err:
                    max_err = err
                    if max_err > 1.0: # Early exit if error is huge
                        return max_err
                        
        return max_err

    # Initial guess: Gauss-Legendre weights
    y_init = np.log(weights + 1e-10)
    
    try:
        from scipy.optimize import minimize
        result = minimize(objective, y_init, method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success and result.fun < 0.1: # Only accept if error is small
            y_opt = result.x
            w_exp = np.exp(y_opt - np.max(y_opt))
            weights_new = w_exp / np.sum(w_exp)
            
            # Sanity check
            if np.all(weights_new > 0) and abs(np.sum(weights_new) - 1.0) < 1e-12:
                weights = weights_new
    except Exception:
        # Fall back to GL weights
        pass

    # Final sanity checks
    if not np.all(np.diff(nodes) > 0):
        nodes = 0.5 * (roots_legendre(8)[0] + 1.0)
        weights = 0.5 * roots_legendre(8)[1]
        
    if not np.all(weights > 0):
        weights = 0.5 * roots_legendre(8)[1]
        
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
