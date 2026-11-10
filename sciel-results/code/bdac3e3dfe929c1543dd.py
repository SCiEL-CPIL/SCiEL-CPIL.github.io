"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    The rule minimizes worst-case relative error across a set of representative
    CPL (Chevallier-Polarski-Linder) cosmologies. 
    """
    # Strict deadline: 60s limit, leave 5s margin for safety
    deadline = time.time() + 55.0
    
    # Representative CPL cosmologies: (w0, wa) pairs
    # We use a diverse set to ensure the rule is robust
    w0_values = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    cosmologies = []
    for w0 in w0_values:
        for wa in wa_values:
            # Skip unphysical combinations if necessary, but CPL is flexible
            cosmologies.append((w0, wa))
            
    Om_m = 0.3
    Om_de = 0.7
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            # Handle potential negative base for non-integer exponent if 1+wa(1-a) < 0
            # For standard CPL in [0,1], 1+wa(1-a) is usually positive for reasonable wa
            base = 1 + wa*(1-a)
            if base <= 0:
                # Fallback to pure power law if base becomes non-positive
                de_term = Om_de * a**(-3*(1+w0+wa))
            else:
                de_term = Om_de * a**(-3*(1+w0+wa)) * base**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral_fast(z, w0, wa, n_ref=200):
        """
        High-accuracy reference integral using Simpson's rule.
        n_ref=200 is fast enough for optimization but accurate enough for surrogate error.
        """
        # Integral from 0 to z of dz'/E(1/(1+z'))
        # Substitution u = z'/z, integral = z * ∫_0^1 du/E(1/(1+zu))
        # Simpson's rule requires odd number of points
        if n_ref % 2 == 0:
            n_ref += 1
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        # Simpson's Rule
        integral = h/3.0 * (integrand[0] + integrand[-1] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-1:2]))
        return z * integral

    def compute_errors(nodes, weights, z_list, cosmo_list):
        """
        Compute relative errors for a given set of nodes/weights.
        Returns a list of relative errors.
        """
        errors = []
        for w0, wa in cosmo_list:
            for z in z_list:
                # Quadrature approximation
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = E_of_a(a_vals, w0, wa)
                quad_val = np.sum(weights / E_vals)
                
                # Reference value
                ref_val = reference_integral_fast(z, w0, wa)
                
                # Relative error
                if ref_val > 1e-10:
                    err = np.abs(quad_val - ref_val) / ref_val
                    errors.append(err)
        return errors

    # Test points: z values covering low to high redshift
    z_test = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Objective function for optimization
    # We want to minimize the worst-case error, but that's non-smooth.
    # We use sum of squared log-errors as a smooth surrogate that penalizes large errors heavily.
    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        # Unpack
        nodes_raw = params[:8]
        weights_raw = params[8:16]
        
        # Parameterize nodes to ensure strictly increasing in (0,1)
        # t_i > 0, nodes = cumsum(t) / sum(t)
        t = np.exp(nodes_raw)
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Parameterize weights to ensure positive and sum to 1
        w_exp = np.exp(weights_raw - np.max(weights_raw))
        weights = w_exp / np.sum(w_exp)
        
        # Calculate errors
        errs = compute_errors(nodes, weights, z_test, cosmologies)
        
        # Sum of squared log-errors (log1p to handle small errors)
        # Scale errors to make log meaningful
        total_error = np.sum(np.log1p(np.array(errs) * 1000)**2)
        
        return total_error

    # Initial guess: Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, gl_weights = roots_legendre(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1)
    
    # Invert node parameterization
    t_init = np.zeros(8)
    t_init[0] = gl_nodes_01[0]
    for i in range(1, 8):
        t_init[i] = gl_nodes_01[i] - gl_nodes_01[i-1]
    nodes_raw_init = np.log(t_init + 1e-10)
    
    # Invert weight parameterization
    weights_raw_init = np.log(gl_weights + 1e-10) - np.log(gl_weights + 1e-10).max()
    
    params_init = np.concatenate([nodes_raw_init, weights_raw_init])
    
    best_params = params_init
    best_score = objective(params_init)
    
    # Optimization Phase 1: L-BFGS-B
    # Reduced iterations to fit within time budget
    try:
        if time.time() < deadline - 5.0:
            result = minimize(objective, params_init, method='L-BFGS-B', 
                              options={'maxiter': 50, 'maxfun': 200, 'disp': False})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
    except Exception:
        pass
        
    # Optimization Phase 2: Nelder-Mead (Global-ish, good for final polish)
    # Nelder-Mead is derivative-free and can handle the non-smoothness of the log-error better
    try:
        if time.time() < deadline - 2.0:
            # Initialize simplex around best_params
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 100, 'maxfev': 200, 'disp': False})
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
    except Exception:
        pass

    # Extract final nodes and weights
    if best_params is not None:
        t = np.exp(best_params[:8])
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        w_exp = np.exp(best_params[8:16] - np.max(best_params[8:16]))
        weights = w_exp / np.sum(w_exp)
    else:
        nodes = gl_nodes_01
        weights = gl_weights

    # Final validation and cleanup
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing
    if not np.all(np.diff(nodes) > 1e-10):
        # Fallback to Gauss-Legendre if optimization broke ordering
        nodes = gl_nodes_01
        weights = gl_weights

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
