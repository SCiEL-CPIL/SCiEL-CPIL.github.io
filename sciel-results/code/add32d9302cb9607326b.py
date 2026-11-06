"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an 8-node positive quadrature rule using Chebyshev-based
    initialization with differential evolution and L-BFGS-B refinement.
    
    Strategy:
    1. Use Chebyshev nodes as initial guess (minimizes Runge phenomenon)
    2. Global search with differential evolution over a diverse cosmology set
    3. Local refinement with L-BFGS-B from best DE solution
    4. Explicit deadline guard for 90s wall-clock limit
    """
    n = 8
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Representative cosmologies: (Omega_m, w0, wa)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.5),    # CPL positive wa
        (0.3, -1.0, -0.5),   # CPL negative wa
        (0.2, -1.0, 0.0),    # Low matter
        (0.4, -1.0, 0.0),    # High matter
        (0.3, -0.8, 0.0),    # Different w0
        (0.3, -1.2, 0.0),    # Different w0
        (0.3, -1.0, 1.0),    # Large wa
        (0.3, -1.0, -1.0),   # Large negative wa
        (0.25, -1.0, 0.3),   # Mixed
    ]
    
    # Redshift test points
    z_tests = np.logspace(-1, 1.5, 12)
    
    def E_z(z, Omega_m, w0, wa):
        """H(z)/H0 for flat CPL model"""
        Omega_de = 1.0 - Omega_m
        z = np.asarray(z, dtype=float)
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_de * (1.0 + z)**(3.0*(1.0 + w0 + wa*z))
        return np.sqrt(term_m + term_de)
    
    def compute_errors(nodes, weights):
        """Compute vector of relative errors over all test cases"""
        from scipy.integrate import quad
        errors = []
        for Omega_m, w0, wa in cosmologies:
            for z_t in z_tests:
                # Quadrature approximation
                a_vals = 1.0 / (1.0 + z_t * nodes)
                E_vals = E_z(z_t * nodes, Omega_m, w0, wa)
                quad_val = np.sum(weights / E_vals)
                
                # Exact integral via adaptive quadrature
                exact, _ = quad(lambda t: 1.0 / E_z(z_t * t, Omega_m, w0, wa), 0, 1, limit=200)
                
                if abs(exact) < 1e-15:
                    errors.append(0.0)
                else:
                    errors.append(abs(quad_val - exact) / abs(exact))
        return np.array(errors)
    
    def objective(params):
        """Minimax objective: worst-case relative error in percent"""
        if time.time() > deadline:
            return np.inf
        
        # Unpack: first n are logit-transformed nodes, next n are log-transformed weights
        node_logits = params[:n]
        weight_logs = params[n:2*n]
        
        nodes = 1.0 / (1.0 + np.exp(-node_logits))
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weight_logs = weight_logs[sort_idx]
        
        if np.any(np.diff(nodes) <= 1e-8):
            return np.inf
        
        weights = np.exp(weight_logs)
        total_w = np.sum(weights)
        if total_w <= 0:
            return np.inf
        weights = weights / total_w
        
        if not np.all(weights > 0):
            return np.inf
        
        errors = compute_errors(nodes, weights)
        return np.max(errors) * 100.0
    
    # Chebyshev nodes on [0,1] as initial guess
    k = np.arange(n)
    cheb_nodes = 0.5 * (1.0 - np.cos((2*k + 1) * np.pi / (2*n)))
    cheb_weights = np.ones(n) / n
    
    # Convert to parameter space
    init_node_logits = np.log(cheb_nodes / (1.0 - cheb_nodes))
    init_weight_logs = np.log(cheb_weights)
    init_params = np.concatenate([init_node_logits, init_weight_logs])
    
    best_params = init_params.copy()
    best_score = objective(init_params)
    
    # Phase 1: Differential Evolution (global search)
    from scipy.optimize import differential_evolution, minimize
    
    # Bounds for logit space (nodes in (0.01, 0.99)) and log weights
    node_bounds = [(-4.0, 4.0)] * n  # logit range for (0.018, 0.982)
    weight_bounds = [(-3.0, 3.0)] * n  # log weight range
    bounds = node_bounds + weight_bounds
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            tol=1e-6,
            popsize=10,
            seed=42,
            init='sobol',
            updating='immediate',
            polish=False,
            callback=lambda x, convergence: None
        )
        if de_result.fun < best_score:
            best_params = de_result.x
            best_score = de_result.fun
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B local refinement from best DE solution
    if time.time() < deadline - 10.0:
        try:
            lbfgs_result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if lbfgs_result.fun < best_score:
                best_params = lbfgs_result.x
                best_score = lbfgs_result.fun
        except Exception:
            pass
    
    # Phase 3: Additional L-BFGS-B restarts from Gauss-Legendre
    if time.time() < deadline - 15.0:
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        gl_nodes = (x_gl + 1.0) / 2.0
        gl_weights = w_gl / 2.0
        gl_weights = gl_weights / np.sum(gl_weights)
        gl_logits = np.log(gl_nodes / (1.0 - gl_nodes))
        gl_wlogs = np.log(gl_weights)
        gl_params = np.concatenate([gl_logits, gl_wlogs])
        
        try:
            lbfgs2 = minimize(
                objective, gl_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if lbfgs2.fun < best_score:
                best_params = lbfgs2.x
                best_score = lbfgs2.fun
        except Exception:
            pass
    
    # Extract final nodes and weights
    node_logits = best_params[:n]
    weight_logs = best_params[n:2*n]
    
    nodes = 1.0 / (1.0 + np.exp(-node_logits))
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weight_logs = weight_logs[sort_idx]
    
    weights = np.exp(weight_logs)
    weights = weights / np.sum(weights)
    
    # Validate
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 1e-8)):
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        weights = weights / np.sum(weights)
    
    if not np.all(weights > 0):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
