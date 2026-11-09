"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, leggauss
from scipy.integrate import quad

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin from 90s

    # 1. Define Cosmology Models
    # We need a representative set of "training" and "validation" cosmologies.
    # Flat universe: Omega_m + Omega_de = 1
    # CPL: w(a) = w0 + wa * (1 - a)
    # H(a)^2 = H0^2 * [Omega_m a^-3 + Omega_de * exp(3 * integral_1^a (1+w(a')) da')]
    
    def h_squared(a, omega_m, w0, wa):
        if a <= 0:
            return np.inf
        # Integral of (1+w(a')) from 1 to a
        # w(a') = w0 + wa(1-a')
        # 1+w(a') = 1 + w0 + wa - wa a'
        # Integral = (1+w0+wa)(a-1) - wa/2 (a^2 - 1)
        # Let's check derivative: (1+w0+wa) - wa*a = 1 + w0 + wa - wa a = 1 + w(a). Correct.
        int_term = (1 + w0 + wa) * (a - 1) - 0.5 * wa * (a**2 - 1)
        omega_de = 1 - omega_m
        de_term = omega_de * np.exp(3 * int_term)
        return omega_m * a**(-3) + de_term

    def integrand(t, z, omega_m, w0, wa):
        # a = 1 / (1 + z*t)
        a = 1.0 / (1.0 + z * t)
        h2 = h_squared(a, omega_m, w0, wa)
        if h2 <= 0:
            return 0.0
        return 1.0 / np.sqrt(h2)

    def exact_lum_dist(z, omega_m, w0, wa):
        # dL/(c/H0) = (1+z) * z * integral_0^1 1/E(a(t)) dt
        # Note: The integral is over t from 0 to 1.
        # Integrand is 1/E(a(t)).
        res, _ = quad(integrand, 0, 1, args=(z, omega_m, w0, wa), limit=50)
        return (1 + z) * z * res

    def quad_lum_dist(z, omega_m, w0, wa, nodes, weights):
        # Sum weights[i] * integrand(nodes[i], ...)
        vals = np.array([integrand(t, z, omega_m, w0, wa) for t in nodes])
        integral_val = np.dot(weights, vals)
        return (1 + z) * z * integral_val

    # 2. Define Parameter Grids
    # Training set for determining weights
    train_params = [
        (0.3, -1.0, 0.0),   # LambdaCDM
        (0.25, -1.0, 0.0),
        (0.3, -0.9, 0.3),   # CPL
        (0.3, -1.1, -0.5),
        (0.2, -1.0, 0.0),
        (0.35, -1.0, 0.0),
    ]
    
    # Validation set for evaluating error
    val_params = [
        (0.3, -1.0, 0.0),
        (0.2, -0.8, 0.5),
        (0.4, -1.2, -0.8),
        (0.25, -0.9, 0.2),
        (0.35, -1.1, -0.3),
        (0.15, -1.0, 0.0),
    ]
    
    # Redshifts to test
    z_values = np.linspace(0.1, 3.0, 10)

    # 3. Optimization Strategy
    # We will optimize the nodes.
    # For a given set of nodes, we determine the optimal weights.
    # We use a least-squares fit to match the integrals of the training set.
    # However, the weights must be positive and sum to 1.
    
    def get_optimal_weights(nodes, train_params_list, z_ref=1.0):
        # Construct matrix A and vector b
        # A[i, j] = integrand(nodes[j], z_ref, params_i)
        # b[i] = exact_integral(z_ref, params_i)
        n_params = len(train_params_list)
        n_nodes = len(nodes)
        A = np.zeros((n_params, n_nodes))
        b = np.zeros(n_params)
        
        for i, (om, w0, wa) in enumerate(train_params_list):
            b[i] = exact_lum_dist(z_ref, om, w0, wa)
            for j in range(n_nodes):
                A[i, j] = integrand(nodes[j], z_ref, om, w0, wa)
        
        # Solve A w = b in least squares sense
        # We also want sum(w) = 1.
        # This is a constrained least squares problem.
        # Simple approach: Solve unconstrained, then project?
        # Or use scipy.optimize.lsq_linear with bounds and equality constraint?
        # lsq_linear doesn't handle equality constraints directly.
        # We can fix one weight or use Lagrange multipliers?
        # Given n_params is small (6) and n_nodes is 8, the system is underdetermined.
        # We want to minimize ||A w - b||^2 subject to w >= 0, sum(w)=1.
        
        # Let's use scipy.optimize.minimize for the weights for each node set?
        # That might be slow inside the node optimization loop.
        
        # Alternative: Use the fact that for a fixed set of nodes, 
        # the "best" weights in a minimax sense are hard to find directly.
        # But we can minimize the maximum error over the validation set directly 
        # by optimizing both nodes and weights together?
        
        # Let's try optimizing nodes and weights together using Nelder-Mead.
        # 16 variables.
        
        return None # Placeholder, will use joint optimization

    def objective(params):
        # params: 8 nodes, 8 weights
        # Unpack
        nodes_raw = params[:8]
        weights_raw = params[8:]
        
        # Enforce node ordering and bounds via transformation?
        # Or just penalize violations.
        # Nodes must be in (0,1) and increasing.
        # Weights must be > 0 and sum to 1.
        
        # Check constraints
        if np.any(nodes_raw <= 0) or np.any(nodes_raw >= 1):
            return 1e6
        if not np.all(np.diff(nodes_raw) > 0):
            return 1e6
        if np.any(weights_raw <= 0):
            return 1e6
        if abs(np.sum(weights_raw) - 1.0) > 1e-4:
            # Penalize sum deviation
            return 1e6 + abs(np.sum(weights_raw) - 1.0)
            
        nodes = nodes_raw
        weights = weights_raw
        
        # Calculate worst-case relative error on validation set
        max_err = 0.0
        for om, w0, wa in val_params:
            for z in z_values:
                exact = exact_lum_dist(z, om, w0, wa)
                approx = quad_lum_dist(z, om, w0, wa, nodes, weights)
                if exact == 0:
                    continue
                rel_err = abs((approx - exact) / exact)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # 4. Initial Guess: Gauss-Legendre
    x, w = leggauss(8)
    nodes_init = (x + 1.0) / 2.0
    weights_init = w / 2.0
    
    # Normalize weights to sum to 1 (they already do, but just in case)
    weights_init = weights_init / np.sum(weights_init)
    
    x0 = np.concatenate([nodes_init, weights_init])
    
    # 5. Optimize
    # Use Nelder-Mead for robustness
    # Time check
    best_params = x0
    best_score = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception as e:
        pass # Keep initial guess if optimization fails

    # Check time
    if time.time() - start_time > time_limit:
        # Return best found so far
        nodes = best_params[:8]
        weights = best_params[8:]
        
        # Final cleanup: ensure strict constraints
        # If nodes are not strictly increasing or out of bounds, fallback to GL
        if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0)):
            nodes = nodes_init
            weights = weights_init
        if not (np.all(weights > 0) and abs(np.sum(weights) - 1.0) < 1e-9):
            # Re-normalize weights if sum is close, else fallback
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                if np.any(weights < 0):
                    nodes = nodes_init
                    weights = weights_init
            else:
                nodes = nodes_init
                weights = weights_init

        return {"nodes": nodes, "weights": weights}

    # Return optimized result
    nodes = best_params[:8]
    weights = best_params[8:]
    
    # Ensure final validity
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0)):
        nodes = nodes_init
        weights = weights_init
    if not (np.all(weights > 0) and abs(np.sum(weights) - 1.0) < 1e-9):
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            if np.any(weights < 0):
                nodes = nodes_init
                weights = weights_init
        else:
            nodes = nodes_init
            weights = weights_init
            
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
