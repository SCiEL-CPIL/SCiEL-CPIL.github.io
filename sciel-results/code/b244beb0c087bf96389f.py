"""
Strategy: Global search using Differential Evolution to minimize worst-case 
relative error over a dense grid of CPL models. The problem is reframed as 
finding nodes and weights that best approximate the integral for a wide 
variety of cosmological parameters, treating the worst-case error as the 
objective function. Vectorization is used to evaluate the objective function 
rapidly, enabling the global optimizer to explore the parameter space 
effectively within the time limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    om_de0 = 1.0 - om_m - om_r - om_k
    # H^2/H0^2 = om_m a^-3 + om_r a^-4 + om_de0 a^(-3(1+w0+wa)) + om_k
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_k + om_de0 * a**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def construct():
    """
    Constructs an 8-node positive quadrature rule on [0,1].
    """
    deadline = time.time() + 85.0  # 5s safety margin
    
    # 1. Define a dense grid of models for optimization
    # We want to cover the likely validation space.
    # z: 0.1 to 8.0
    # w0, wa: -2.0 to 2.0
    # om_m: 0.2 to 0.4
    # om_r: fixed at 9e-5 (negligible for z > 0.1, but included for completeness)
    # om_k: 0.0 (flat universe)
    
    # Create a grid of challenging models
    # We use a stratified sampling approach to ensure coverage.
    z_vals = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0])
    w0_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    wa_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    om_m_vals = np.array([0.2, 0.3, 0.4])
    
    # Combine into a list of models
    # To keep the objective function fast, we limit the number of models 
    # but ensure they are representative. 
    # A full grid would be 7*9*5*3 = 945 models. This is manageable with vectorization.
    models = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for om_m in om_m_vals:
                    models.append((z, w0, wa, om_m, 9e-5, 0.0))
    
    # Add some random models to break symmetry and catch edge cases
    np.random.seed(42)
    for _ in range(100):
        z = np.random.uniform(0.1, 8.0)
        w0 = np.random.uniform(-2.0, 2.0)
        wa = np.random.uniform(-2.0, 2.0)
        om_m = np.random.uniform(0.2, 0.4)
        models.append((z, w0, wa, om_m, 9e-5, 0.0))
        
    # Convert to numpy arrays for vectorization
    # models is a list of tuples. Let's stack them.
    # z, w0, wa, om_m, om_r, om_k
    z_arr = np.array([m[0] for m in models])
    w0_arr = np.array([m[1] for m in models])
    wa_arr = np.array([m[2] for m in models])
    om_m_arr = np.array([m[3] for m in models])
    om_r_arr = np.array([m[4] for m in models])
    om_k_arr = np.array([m[5] for m in models])
    
    n_models = len(z_arr)
    
    # Precompute high-precision reference values
    # Using 64-point Gauss-Legendre for reference. 
    # 64 points is usually sufficient for smooth functions like 1/E(a) 
    # to achieve relative error < 1e-10, which is well below the quadrature error.
    gl_nodes_ref, gl_weights_ref = np.polynomial.legendre.leggauss(64)
    gl_nodes_ref = (gl_nodes_ref + 1.0) / 2.0
    gl_weights_ref = gl_weights_ref / 2.0
    
    # Reference integral values for each model
    # I_true[j] = integral for model j
    I_true = np.zeros(n_models)
    for j in range(n_models):
        a_gl = 1.0 / (1.0 + z_arr[j] * gl_nodes_ref)
        E_gl = _E_cpl(a_gl, w0_arr[j], wa_arr[j], om_m_arr[j], om_r_arr[j], om_k_arr[j])
        f_gl = 1.0 / E_gl
        I_true[j] = np.dot(f_gl, gl_weights_ref)
        
    # Ensure I_true is not zero (shouldn't happen for z>0)
    I_true = np.maximum(I_true, 1e-10)
    
    def objective(params):
        """
        params: 16-dimensional vector.
        First 8: nodes (sorted, in (0,1))
        Last 8: log-weights (to ensure positivity)
        """
        nodes = params[:8]
        log_weights = params[8:]
        
        # Check validity
        if not np.all(np.diff(nodes) > 1e-8):
            return 1e10
        if nodes[0] <= 0.0 or nodes[-1] >= 1.0:
            return 1e10
            
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        # Compute quadrature values for all models
        # a_nodes[i, j] = 1 / (1 + z_j * nodes_i)
        # Shape: (8, n_models)
        a_nodes = 1.0 / (1.0 + z_arr[None, :] * nodes[:, None])
        
        # E_nodes[i, j] = E(a_nodes[i, j])
        # We need to broadcast w0, wa, etc.
        # w0_arr shape: (n_models,)
        # a_nodes shape: (8, n_models)
        
        E_nodes = _E_cpl(a_nodes, 
                         w0_arr[None, :], 
                         wa_arr[None, :], 
                         om_m_arr[None, :], 
                         om_r_arr[None, :], 
                         om_k_arr[None, :])
                         
        f_nodes = 1.0 / E_nodes
        
        # Q_val[j] = sum_i weights[i] * f_nodes[i, j]
        Q_val = f_nodes.T @ weights
        
        # Relative error
        rel_err = np.abs(Q_val - I_true) / I_true
        
        # Worst-case error
        max_err = np.max(rel_err)
        
        return max_err

    # 2. Differential Evolution Setup
    # Bounds for nodes: (1e-6, 1-1e-6)
    # Bounds for log_weights: (-20, 20) -> weights in [2e-9, 4.8e8]
    # The normalization handles the scale, so the bounds just need to be wide enough.
    
    bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(-20.0, 20.0)] * 8
    
    # Differential Evolution options
    # popsize: 15 * 16 = 240 individuals.
    # maxiter: 50.
    # Tolerance: 1e-6.
    # Seed: 42 for reproducibility.
    
    # We need to be careful with the time limit.
    # DE is stochastic and can be slow.
    # Let's run it and check time.
    
    start_time = time.time()
    
    # Use a custom callback to stop if time is up
    def callback(xk, convergence):
        return time.time() > deadline - 10.0
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            mutation=(0.5, 1.5), 
            recombination=0.7, 
            polish=False, # Disable Nelder-Mead polish to save time and avoid local minima traps
            callback=callback,
            update_interval=1
        )
        
        best_params = res.x
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to Gauss-Legendre
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        gl_nodes = (gl_nodes + 1.0) / 2.0
        gl_weights = gl_weights / 2.0
        return {
            "nodes": gl_nodes,
            "weights": gl_weights
        }
        
    # 3. Final Polish / Validation
    best_nodes = best_params[:8]
    log_weights = best_params[8:]
    best_weights = np.exp(log_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure strict increasing order and bounds
    if not (np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(np.diff(best_nodes) > 1e-6)):
        # If invalid, fallback to GL
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        gl_nodes = (gl_nodes + 1.0) / 2.0
        gl_weights = gl_weights / 2.0
        best_nodes = gl_nodes
        best_weights = gl_weights
    else:
        # Sort nodes and corresponding weights just in case DE messed up order (unlikely with bounds, but safe)
        # Actually, DE optimizes the vector, so if it finds a permutation that is better, 
        # the nodes might not be sorted. But the problem requires strictly increasing nodes.
        # The objective function penalizes unsorted nodes heavily (returns 1e10).
        # So if it converged, nodes should be sorted. 
        # However, if we want to be absolutely safe, we can sort them.
        # Sorting changes the rule, so we must re-evaluate error? 
        # No, the rule is defined by the set of (node, weight) pairs. 
        # If we sort nodes, we must sort weights accordingly to keep the pairs intact.
        sort_idx = np.argsort(best_nodes)
        best_nodes = best_nodes[sort_idx]
        best_weights = best_weights[sort_idx]
        
        # Re-check validity after sorting
        if not (np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(np.diff(best_nodes) > 1e-6)):
            gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
            gl_nodes = (gl_nodes + 1.0) / 2.0
            gl_weights = gl_weights / 2.0
            best_nodes = gl_nodes
            best_weights = gl_weights

    # Final normalization check
    best_weights = best_weights / np.sum(best_weights)
    
    return {
        "nodes": best_nodes,
        "weights": best_weights
    }
# EVOLVE-BLOCK-END
