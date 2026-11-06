"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.integrate import quad

def E_of_a(a, Omega_m, w0, w1):
    """Compute H(a)/H0 for CPL parametrization in a flat universe."""
    a = np.asarray(a)
    a_safe = np.maximum(a, 1e-30)
    ln_a = np.log(a_safe)
    exponent = 3.0 * (1.0 + w0) * ln_a + 3.0 * w1 * (a_safe - 1.0 - ln_a)
    de_term = (1.0 - Omega_m) * np.exp(exponent)
    m_term = Omega_m
    return np.sqrt(a_safe ** (-3.0) * (m_term + de_term))

def compute_reference_value(z, Omega_m, w0, w1, num_points=128):
    """Compute the reference integral using high-order Gauss-Legendre."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(num_points)
    x_mapped = (x_gl + 1.0) / 2.0
    a_vals = 1.0 / (1.0 + z * x_mapped)
    integrand_vals = 1.0 / E_of_a(a_vals, Omega_m, w0, w1)
    return np.sum(w_gl * integrand_vals) / 2.0

def compute_worst_error(nodes, weights, cosmologies, z_values):
    """Compute the worst-case relative error."""
    max_err = 0.0
    for (Omega_m, w0, w1) in cosmologies:
        for z in z_values:
            # Quadrature approximation
            a_quad = 1.0 / (1.0 + z * nodes)
            quad_val = np.sum(weights / E_of_a(a_quad, Omega_m, w0, w1))
            
            # Reference value
            ref_val = compute_reference_value(z, Omega_m, w0, w1)
            
            if abs(ref_val) < 1e-30:
                continue
            err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Representative cosmologies
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -0.8, 0.3),
        (0.3, -1.2, 0.5),
        (0.3, -0.9, -0.3),
        (0.25, -1.0, 0.2),
        (0.35, -1.1, 0.4),
        (0.3, -0.7, 0.6),
    ]
    
    # Redshift values
    z_values = np.logspace(-1, 2, 8)
    
    # Initial guess: Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = (x_gl + 1.0) / 2.0
    weights_init = w_gl / 2.0
    
    # Parameterization for Differential Evolution
    # Nodes: use gaps to ensure increasing order
    # Let u be 9 parameters (8 gaps + 1 end gap? No, 8 nodes need 9 segments: 0->x0, x0->x1, ..., x7->1)
    # Let gaps g_i = exp(u_i) for i=0..8
    # x_i = (sum_{j=0}^i g_j) / (sum_{k=0}^8 g_k)
    # Weights: let v be 8 parameters, w_i = exp(v_i) / sum(exp(v))
    
    def decode_params(params):
        # params: 9 node gap params + 8 weight params
        node_gaps = np.exp(params[:9])
        total_gap = np.sum(node_gaps)
        nodes = np.cumsum(node_gaps) / total_gap
        # The last node will be < 1, which is correct.
        # We need exactly 8 nodes.
        # cumsum of 9 elements gives 9 points. The 9th is 1.0.
        # We take the first 8.
        nodes_8 = nodes[:8]
        
        weight_params = params[9:]
        weights_raw = np.exp(weight_params)
        weights = weights_raw / np.sum(weights_raw)
        
        return nodes_8, weights

    def objective(params):
        if time.time() > deadline:
            return 1e100 # Penalty for timeout
        nodes, weights = decode_params(params)
        # Check for numerical issues
        if not np.all(np.diff(nodes) > 1e-12):
            return 1e100
        if not np.all(weights > 0):
            return 1e100
        
        err = compute_worst_error(nodes, weights, cosmologies, z_values)
        return err

    # Initial parameters
    # We want to start near Gauss-Legendre nodes
    # Invert the gap parameterization?
    # It's easier to just provide reasonable random bounds and let DE explore.
    # But we can try to initialize close to GL.
    # GL nodes in [0,1]: ~0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95
    # Gaps: 0.05, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.05
    # log(gaps) ~ -3, -1.9, -1.9, ...
    
    # Bounds for DE
    # u_i (gaps): log(gap). Gaps can be small or large.
    # Let's bound u_i in [-10, 10]
    # v_i (weights): log(weight). Weights sum to 1, so each is < 1.
    # Let's bound v_i in [-10, 10]
    
    bounds = [(-10, 10)] * 9 + [(-10, 10)] * 8
    
    # Initial population seed
    # We can construct a seed based on GL nodes
    # gaps_gl = np.diff(np.concatenate([[0], nodes_init, [1]]))
    # u_init = np.log(gaps_gl)
    # v_init = np.log(weights_init)
    # seed = np.concatenate([u_init, v_init])
    
    # However, DE uses a population. We can just let it generate random ones in bounds.
    # To be safe and fast, we'll use a small population and few iterations.
    
    try:
        # DE is stochastic, we set seed for reproducibility if needed, but random is fine.
        # maxiter=50, popsize=15
        # This is 750 evaluations.
        # Each eval is ~1ms. Total ~1s.
        # We can afford more.
        # maxiter=100, popsize=20 -> 2000 evals ~ 2s.
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False # Skip local polish to save time, DE is global
        )
        
        if time.time() > deadline:
            return {"nodes": nodes_init, "weights": weights_init}
        
        best_params = result.x
        nodes_best, weights_best = decode_params(best_params)
        
        # Verify constraints
        if not np.all(np.diff(nodes_best) > 1e-12):
            return {"nodes": nodes_init, "weights": weights_init}
        
        # Normalize weights just in case
        weights_best = weights_best / np.sum(weights_best)
        
        # Final check: compute error of best vs initial
        err_best = compute_worst_error(nodes_best, weights_best, cosmologies, z_values)
        err_init = compute_worst_error(nodes_init, weights_init, cosmologies, z_values)
        
        if err_best < err_init:
            return {"nodes": nodes_best, "weights": weights_best}
        else:
            return {"nodes": nodes_init, "weights": weights_init}
            
    except Exception as e:
        return {"nodes": nodes_init, "weights": weights_init}

# EVOLVE-BLOCK-END
