import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Define E(a) for CPL dark energy in flat universe
    # E(a) = sqrt(Omega_m * a^-3 + Omega_de * a^-3(1+w0) * exp(3*w1*(1-a)/a))
    # Note: The task formula uses E(a) = H(a)/H0.
    # Standard CPL: w(a) = w0 + w1*(1-a)
    # H^2(a) = H0^2 [Omega_m a^-3 + Omega_de a^-3 exp(3 ∫_a^1 w(a')/a' da')]
    # ∫_a^1 (w0 + w1(1-a'))/a' da' = w0 ln(1/a) + w1 (1/a - 1)
    # So exponent is 3 * [ -w0 ln(a) + w1 (1/a - 1) ] = -3 w0 ln(a) + 3 w1 (1/a - 1)
    # a^-3 * exp(3 w1 (1/a - 1)) * a^(3 w0) = a^(-3(1-w0)) * exp(3 w1 (1/a - 1))
    # Wait, let's re-verify the standard form.
    # H^2/H0^2 = Omega_m a^-3 + Omega_de a^-3(1+w0) exp(3 w1 (1-a)/a)
    # This matches the standard CPL form used in many codes.
    
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        # Term: a^-3(1+w0) * exp(3*w1*(1-a)/a)
        # (1-a)/a = 1/a - 1
        exp_term = np.exp(3.0 * w1 * (1.0/a_safe - 1.0))
        de_term = Omega_de * a_safe**(-3.0*(1.0+w0)) * exp_term
        m_term = Omega_m * a_safe**(-3.0)
        return np.sqrt(m_term + de_term)

    def _true_distance(z_max, w0, w1):
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) / _E(a, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val

    # Test cosmologies: diverse CPL models
    # We want to cover a range of w0 and w1 to ensure robustness.
    # Typical values: w0 in [-1.5, -0.5], w1 in [-1.0, 1.0]
    test_cosmos = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL w1>0
        (-1.0, -0.5),     # CPL w1<0
        (-0.9, 0.3),      # Quintessence-like
        (-0.95, -0.3),    # Another variant
        (-1.2, 0.0),      # Phantom
        (-0.8, 0.5),      # Strong evolution
        (-1.0, 1.0),      # Strong CPL
        (-1.1, 0.4),      # Additional
        (-0.85, -0.4),    # Additional
        (-0.7, 0.8),      # Very strong evolution
        (-1.3, -0.2),     # Phantom with slight evolution
        (-1.0, 0.8),      # Strong positive w1
        (-1.0, -0.8),     # Strong negative w1
        (-0.5, 0.0),      # Stiffer DE
        (-1.5, 0.0),      # Softer DE
    ]
    
    # Redshifts to test
    test_zs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # Pre-compute true distances for speed
    # Shape: (N_cosmo, N_z)
    true_vals = np.zeros((len(test_cosmos), len(test_zs)))
    for i, (w0, w1) in enumerate(test_cosmos):
        for j, z in enumerate(test_zs):
            true_vals[i, j] = _true_distance(z, w0, w1)

    # Vectorized quadrature distance calculation
    # nodes: shape (8,)
    # weights: shape (8,)
    # w0, w1: scalars or arrays
    # z_max: scalar or array
    
    def _quad_distance_vectorized(nodes, weights, w0_arr, w1_arr, z_arr):
        """
        Compute quadrature distances for all combinations of (w0, w1) and z.
        w0_arr, w1_arr: shape (N_cosmo,)
        z_arr: shape (N_z,)
        Returns: shape (N_cosmo, N_z)
        """
        # z_nodes = z_max * nodes
        # z_max shape: (N_z, 1)
        # nodes shape: (1, 8)
        z_nodes = z_arr[:, np.newaxis] * nodes[np.newaxis, :] # (N_z, 8)
        
        # a_nodes = 1 / (1 + z_nodes)
        a_nodes = 1.0 / (1.0 + z_nodes) # (N_z, 8)
        
        # We need E(a_nodes) for each cosmology.
        # w0_arr shape: (N_cosmo, 1, 1)
        # w1_arr shape: (N_cosmo, 1, 1)
        w0_b = w0_arr[:, np.newaxis, np.newaxis]
        w1_b = w1_arr[:, np.newaxis, np.newaxis]
        
        # a_nodes shape: (1, N_z, 8)
        a_b = a_nodes[np.newaxis, :, :]
        
        # Calculate E(a)
        a_safe = np.maximum(a_b, 1e-10)
        exp_term = np.exp(3.0 * w1_b * (1.0/a_safe - 1.0))
        de_term = (1.0 - 0.3) * a_safe**(-3.0*(1.0+w0_b)) * exp_term
        m_term = 0.3 * a_safe**(-3.0)
        E_vals = np.sqrt(m_term + de_term) # (N_cosmo, N_z, 8)
        
        # Integrand values: (1+z) / E(a)
        # 1+z_nodes shape: (1, N_z, 8)
        integrand_vals = (1.0 + z_nodes)[np.newaxis, :, :] / E_vals # (N_cosmo, N_z, 8)
        
        # Sum over nodes with weights
        # weights shape: (1, 1, 8)
        w_b = weights[np.newaxis, np.newaxis, :]
        sum_term = np.sum(integrand_vals * w_b, axis=2) # (N_cosmo, N_z)
        
        # Final distance: z_max * sum
        # z_arr shape: (N_z,) -> (1, N_z)
        distances = z_arr[np.newaxis, :] * sum_term
        
        return distances

    def _objective(params):
        if time.time() > deadline - 2:
            return 1e10
            
        # Unpack parameters
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Transform nodes: cumsum(exp(u)) / total
        # This ensures strictly increasing nodes in (0, 1)
        exp_u = np.exp(node_params)
        cumsum_u = np.cumsum(exp_u)
        nodes = cumsum_u / cumsum_u[-1]
        
        # Transform weights: softmax
        # This ensures positive weights summing to 1
        exp_w = np.exp(weight_params - np.max(weight_params)) # numerical stability
        weights = exp_w / np.sum(exp_w)
        
        # Compute distances
        # w0_arr, w1_arr from test_cosmos
        w0_arr = np.array([c[0] for c in test_cosmos])
        w1_arr = np.array([c[1] for c in test_cosmos])
        
        try:
            approx_vals = _quad_distance_vectorized(nodes, weights, w0_arr, w1_arr, np.array(test_zs))
        except Exception:
            return 1e10
            
        # Calculate relative errors
        rel_errors = np.abs(approx_vals - true_vals) / true_vals
        
        # Worst case error in percent
        max_err = np.max(rel_errors) * 100.0
        
        return max_err

    # Differential Evolution
    # Bounds for node_params: [-5, 5] is usually enough for exp
    # Bounds for weight_params: [-5, 5] is enough for softmax
    bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
    
    # Run DE
    # maxiter and popsize can be adjusted based on time
    # We have ~85s. DE is stochastic.
    # Let's try to run DE for a significant portion of the time.
    
    # Monitor time
    t_start = time.time()
    
    # Use a callback to stop if time is running out
    def callback(xk, convergence=True):
        return time.time() > deadline - 15.0

    # Run DE
    # tol and atol for DE
    de_result = differential_evolution(_objective, bounds, seed=42, maxiter=50, popsize=20, 
                                       tol=1e-8, atol=1e-8, callback=callback, polish=False)
    
    best_params = de_result.x
    best_score = de_result.fun
    
    # Local Polish with Nelder-Mead or L-BFGS-B
    # L-BFGS-B might struggle with the non-smooth max function, but Nelder-Mead is derivative-free.
    # Let's try Nelder-Mead for the final polish.
    
    def _polish_objective(params):
        if time.time() > deadline - 5:
            return 1e10
        return _objective(params)

    # Use the best DE result as starting point
    try:
        nm_result = minimize(_polish_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if nm_result.fun < best_score:
            best_params = nm_result.x
            best_score = nm_result.fun
    except Exception:
        pass

    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    exp_u = np.exp(node_params)
    cumsum_u = np.cumsum(exp_u)
    nodes = cumsum_u / cumsum_u[-1]
    
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Ensure strict constraints
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if nodes[j] <= nodes[j-1]:
            nodes[j] = nodes[j-1] + 1e-8
    
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
