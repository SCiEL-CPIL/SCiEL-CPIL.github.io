"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin

    # 1. Define representative cosmologies and redshifts
    # Cosmologies: (Omega_m, Omega_r, Omega_L, w0, wa)
    # Flat universe: Omega_k = 0
    cosmologies = [
        (0.3, 1e-4, 0.7, -1.0, 0.0),    # LCDM
        (0.25, 1e-4, 0.75, -1.0, 0.0),  # LCDM variant
        (0.3, 1e-4, 0.7, -0.9, 0.5),    # CPL
        (0.3, 1e-4, 0.7, -1.1, 0.5),    # CPL
        (0.2, 1e-4, 0.8, -1.0, 0.0),    # Different Omega_m
        (0.3, 1e-4, 0.7, -1.0, 1.0),    # High wa
        (0.3, 1e-4, 0.7, -1.5, 0.0),    # Phantom
        (0.3, 1e-4, 0.7, -0.5, -0.5),   # Quintessence
    ]
    
    # Redshifts to test
    z_values = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    
    # Pre-compute exact integrals for the grid
    # Integral I = int_0^1 (1 / E(z*u)) du
    # E(a) = sqrt(Om*a^-3 + Or*a^-4 + OL + Ox*a^(-3*(1+w0+wa)))
    # a = 1/(1+z*u)
    
    exact_integrals = []
    integrands_at_nodes = [] # This will be filled during optimization? No, depends on nodes.
    
    def E_of_a(a, Om, Or, OL, Ox, w0, wa):
        # Avoid division by zero or negative values under sqrt
        # a is in (0, 1]
        term_m = Om * a**(-3)
        term_r = Or * a**(-4)
        term_l = OL
        term_x = Ox * a**(-3 * (1 + w0 + wa))
        val = term_m + term_r + term_l + term_x
        if val < 0:
            return np.inf
        return np.sqrt(val)
    
    def integrand_u(u, z, Om, Or, OL, Ox, w0, wa):
        a = 1.0 / (1.0 + z * u)
        E = E_of_a(a, Om, Or, OL, Ox, w0, wa)
        if np.isinf(E):
            return 0.0
        return 1.0 / E
    
    # Pre-compute exact values
    for Om, Or, OL, w0, wa in cosmologies:
        Ox = 1.0 - Om - Or - OL
        for z in z_values:
            # Integrate from 0 to 1
            # quad is adaptive, should be accurate
            try:
                val, err = quad(integrand_u, 0, 1, args=(z, Om, Or, OL, Ox, w0, wa), limit=100)
                exact_integrals.append(val)
            except Exception:
                exact_integrals.append(1.0) # Fallback

    exact_integrals = np.array(exact_integrals)
    n_grid = len(exact_integrals)
    
    # 2. Optimization
    # Parameters: 16 variables
    # First 8: log-spaced node parameters to ensure 0 < n1 < ... < n8 < 1
    # Let y_i be the parameters.
    # n_i = sigmoid(y_i) is not enough for ordering.
    # Use cumulative softmax or similar.
    # Simpler: n_i = (i + offset) / (8 + 1) is fixed. 
    # We want to optimize nodes too.
    # Representation: 
    # Let t_i be free variables.
    # n_i = 1 / (1 + exp(-t_i)) ensures (0,1) but not ordering.
    # To ensure ordering, we can optimize sorted differences.
    # Let d_1 = n_1, d_i = n_i - n_{i-1}.
    # d_i > 0.
    # Let d_i = exp(v_i).
    # Then n_i = sum_{j=1}^i exp(v_j).
    # To ensure n_8 < 1, we can scale: n_i = (sum_{j=1}^i exp(v_j)) / (sum_{k=1}^8 exp(v_k) + 1) ?
    # Actually, if we just normalize the cumulative sum to sum to 1, we get points in [0,1].
    # Let S_i = sum_{j=1}^i exp(v_j).
    # n_i = S_i / S_8. This gives n_8 = 1. We need n_8 < 1.
    # Let's add a dummy variable v_9 for the gap after n_8.
    # Let v = [v_1, ..., v_9].
    # Gaps g_i = exp(v_i).
    # Total gap G = sum_{i=1}^9 g_i.
    # n_i = (sum_{j=1}^i g_j) / G.
    # This ensures 0 < n_1 < ... < n_8 < 1.
    
    # Weights:
    # w_i = exp(u_i) / sum(exp(u_j)).
    
    def decode_params(params):
        v = params[:9]
        u = params[9:]
        
        g = np.exp(v)
        S = np.cumsum(g)
        G = S[-1]
        nodes = S[:8] / G
        
        w_raw = np.exp(u)
        weights = w_raw / np.sum(w_raw)
        
        return nodes, weights

    def objective(params):
        nodes, weights = decode_params(params)
        
        # Calculate quadrature estimates for all grid points
        # We need to evaluate g(n_i) for each grid point.
        # g(u, z, ...) = 1/E(z*u)
        
        total_error = 0.0
        max_rel_err = 0.0
        
        # Vectorize node evaluation if possible?
        # nodes shape (8,)
        # For each grid point k (z_k, cosmo_k):
        #   a_i = 1 / (1 + z_k * nodes)
        #   E_i = E(a_i)
        #   I_quad = sum(weights * 1/E_i)
        
        idx = 0
        for Om, Or, OL, w0, wa in cosmologies:
            Ox = 1.0 - Om - Or - OL
            for z in z_values:
                a_vals = 1.0 / (1.0 + z * nodes)
                # Compute E for all 8 nodes at once
                term_m = Om * a_vals**(-3)
                term_r = Or * a_vals**(-4)
                term_l = OL
                term_x = Ox * a_vals**(-3 * (1 + w0 + wa))
                E_vals = np.sqrt(term_m + term_r + term_l + term_x)
                
                # Avoid division by zero
                inv_E = 1.0 / E_vals
                I_quad = np.dot(weights, inv_E)
                
                I_exact = exact_integrals[idx]
                
                if I_exact == 0:
                    rel_err = abs(I_quad)
                else:
                    rel_err = abs(I_quad - I_exact) / I_exact
                
                total_error += rel_err**2
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                
                idx += 1
        
        # Return a combination of mean squared error and max error to balance
        # Minimizing max error directly can be non-smooth.
        # Let's minimize total_error + lambda * max_rel_err
        return total_error + 10.0 * max_rel_err

    # Initial guess
    # Nodes: uniform
    # Weights: uniform
    # v: log(gaps). Uniform gaps => v_i = 0
    # u: log(weights). Uniform weights => u_i = 0
    x0 = np.zeros(17)
    
    # Optimize
    best_params = x0
    best_score = objective(x0)
    
    # Try a few optimization methods or iterations
    # Use Powell as it doesn't need gradients and is robust to non-smoothness
    try:
        result = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'disp': False})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # If close to deadline, return current best
        pass
    else:
        # Try L-BFGS-B with gradient? No gradient provided. 
        # Maybe try Nelder-Mead?
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 500, 'disp': False})
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
        except Exception:
            pass

    nodes, weights = decode_params(best_params)
    
    # Final validation and cleanup
    # Ensure strictly increasing
    # The decoding ensures this, but check for numerical issues
    if not np.all(np.diff(nodes) > 0):
        # Fallback to Gauss-Legendre
        from numpy.polynomial.legendre import leggauss
        x, w = leggauss(8)
        nodes = (x + 1.0) / 2.0
        weights = w / 2.0
    else:
        # Normalize weights to sum to 1 exactly
        weights = weights / np.sum(weights)
        
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
