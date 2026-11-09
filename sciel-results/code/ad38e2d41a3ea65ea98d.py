"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Representative cosmologies: (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (w1 positive)
        (0.3, -1.0, -0.5),  # Evolving DE (w1 negative)
        (0.2, -1.0, 0.0),   # Low matter density
        (0.4, -1.0, 0.0),   # High matter density
        (0.3, -1.5, 0.3),   # Strongly evolving
        (0.3, -0.5, -0.3),  # Mild quintessence with evolution
        (0.35, -1.1, 0.2),  # Slightly matter-dominated
    ]
    
    # Test redshifts - broader range for better generalization
    z_test = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5])
    
    def E_func(z, Om, w0, w1):
        OL = 1.0 - Om
        one_plus_z = 1.0 + z
        # E^2 = Om(1+z)^3 + OL(1+z)^(3(1+w0+w1*z))
        return np.sqrt(Om * one_plus_z**3 + OL * one_plus_z**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Om, w0, w1):
        """Compute the true integral using adaptive quadrature."""
        def integrand(u):
            return 1.0 / E_func(z * u, Om, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
        return res
    
    # Precompute true integrals for all test cases
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
    
    # Check time after precomputation
    if time.time() > deadline:
        # Fallback to Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}
    
    # Precompute E values for all cosmo, z, and node positions for fast evaluation
    # We'll use a vectorized approach in the objective
    
    def compute_nodes_from_gaps(gap_params):
        """
        Convert gap parameters to nodes.
        gap_params: 7 values, each in (0, inf) via log transform
        nodes[0] = exp(gap_params[0]) / (1 + exp(gap_params[0]))  # first node position
        Actually, let's use a simpler approach:
        gaps are the differences between consecutive nodes
        node[0] = first_node
        node[i] = node[i-1] + gap[i]
        
        Better: use 8 parameters where:
        - 7 gap parameters (log-spaced, positive)
        - 1 offset parameter for the first node
        
        Let's use: params = [offset, gap1_log, gap2_log, ..., gap7_log]
        offset in (0, 0.5) via sigmoid
        gap_i = exp(gap_log_i) * scale_factor
        Then normalize so nodes are in (0, 1)
        """
        offset = 1.0 / (1.0 + np.exp(-gap_params[0])) * 0.4 + 0.02  # in (0.02, 0.42)
        log_gaps = gap_params[1:]
        gaps = np.exp(log_gaps) * 0.1  # scale factor
        
        # Build nodes
        nodes = np.zeros(8)
        nodes[0] = offset
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + gaps[i-1]
        
        # Normalize to [0, 1] with small margin
        if nodes[-1] > 0.98:
            scale = 0.98 / nodes[-1]
            nodes = nodes * scale
        
        # Ensure strict increase and within (0, 1)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
        
        return nodes
    
    def compute_weights_from_params(weight_params):
        """Convert weight parameters to positive weights summing to 1."""
        # Use softmax for stability
        shifted = weight_params - np.max(weight_params)
        exp_w = np.exp(shifted)
        weights = exp_w / np.sum(exp_w)
        return weights
    
    def objective(params):
        """
        params: 16 floats
        first 8: [offset, log_gap1, ..., log_gap7]
        last 8: weight parameters
        """
        if time.time() > deadline:
            return 1.0  # Large penalty
        
        node_params = params[:8]
        weight_params = params[8:]
        
        nodes = compute_nodes_from_gaps(node_params)
        weights = compute_weights_from_params(weight_params)
        
        # Vectorized evaluation of quadrature for all cosmo and z
        # nodes shape: (8,)
        # For each cosmo i and z j, compute sum(w_k / E(z_j * nodes_k))
        
        max_err = 0.0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_nodes = z * nodes  # shape (8,)
                E_vals = E_func(z_nodes, Om, w0, w1)  # shape (8,)
                integrand_vals = 1.0 / E_vals  # shape (8,)
                quad_val = np.dot(weights, integrand_vals)
                
                true_val = true_vals[i, j]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
                        
        return max_err
    
    # Generate multiple initial points for multi-start optimization
    initial_points = []
    
    # 1. Gauss-Legendre
    n = 8
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    gl_weights = 0.5 * gl_weights
    gl_weights = gl_weights / gl_weights.sum()
    
    # Invert to get gap parameters (approximate)
    gl_gaps = np.diff(gl_nodes)
    gl_offset = gl_nodes[0]
    # Convert to log scale
    gl_log_gaps = np.log(gl_gaps / 0.1)
    gl_offset_param = np.log(gl_offset / 0.4 * (1.0 - np.exp(0)) / np.exp(0))  # approximate
    # Simpler: just use the offset directly
    initial_points.append(np.concatenate([[np.log(gl_offset / 0.4 * 0.98)], gl_log_gaps, np.log(gl_weights / gl_weights.mean())]))
    
    # 2. Chebyshev nodes
    cheb_nodes = 0.5 * (1.0 - np.cos(np.pi * (np.arange(1, 9) - 0.5) / 8))
    cheb_weights = np.ones(8) / 8
    cheb_gaps = np.diff(cheb_nodes)
    cheb_offset = cheb_nodes[0]
    cheb_log_gaps = np.log(cheb_gaps / 0.1)
    initial_points.append(np.concatenate([[np.log(cheb_offset / 0.4 * 0.98)], cheb_log_gaps, np.zeros(8)]))
    
    # 3. Uniform
    uni_nodes = np.linspace(0.05, 0.95, 8)
    uni_weights = np.ones(8) / 8
    uni_gaps = np.diff(uni_nodes)
    uni_offset = uni_nodes[0]
    uni_log_gaps = np.log(uni_gaps / 0.1)
    initial_points.append(np.concatenate([[np.log(uni_offset / 0.4 * 0.98)], uni_log_gaps, np.zeros(8)]))
    
    # 4. Clenshaw-Curtis-like (cosine spacing)
    cc_nodes = 0.5 * (1.0 - np.cos(np.pi * np.arange(8) / 7))
    cc_weights = np.ones(8) / 8
    cc_gaps = np.diff(cc_nodes)
    cc_offset = cc_nodes[0]
    cc_log_gaps = np.log(cc_gaps / 0.1)
    initial_points.append(np.concatenate([[np.log(cc_offset / 0.4 * 0.98)], cc_log_gaps, np.zeros(8)]))
    
    # 5. Slightly perturbed Gauss-Legendre
    for pert in [0.01, -0.01, 0.05]:
        pert_nodes = gl_nodes + pert * np.sin(np.pi * gl_nodes)
        pert_nodes = np.clip(pert_nodes, 0.01, 0.99)
        pert_gaps = np.diff(pert_nodes)
        pert_offset = pert_nodes[0]
        pert_log_gaps = np.log(pert_gaps / 0.1)
        initial_points.append(np.concatenate([[np.log(pert_offset / 0.4 * 0.98)], pert_log_gaps, np.log(gl_weights / gl_weights.mean())]))
    
    # Run multi-start L-BFGS-B optimization
    best_result = None
    best_obj = np.inf
    
    for init_idx, init_params in enumerate(initial_points):
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue
    
    # If no result found, fallback to Gauss-Legendre
    if best_result is None:
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    # Extract best params
    best_params = best_result.x
    node_params = best_params[:8]
    weight_params = best_params[8:]
    
    # Reconstruct nodes and weights
    nodes = compute_nodes_from_gaps(node_params)
    weights = compute_weights_from_params(weight_params)
    
    # Final validation
    if not (len(nodes) == 8 and len(weights) == 8):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(np.diff(nodes) > 0):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(weights > 0):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
