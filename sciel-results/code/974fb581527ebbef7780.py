"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-15)
    exponent = wa / (1.0 + w0)
    return a**(-3.0) * base**exponent


def _exact_integral_closed_form(z, w0, wa):
    """
    Closed-form integral_0^z dt/E(t) for CPL model.
    E(t) = (1+t)^3 * (1+w0+wa*t)^(wa/(1+w0))
    1/E(t) = (1+t)^(-3) * (1+w0+wa*t)^(-wa/(1+w0))
    Let beta = wa/(1+w0), gamma = 1+w0
    integral_0^z (1+t)^(-3) * (gamma + wa*t)^(-beta) dt
    
    Use substitution u = 1+t, so t = u-1, dt = du
    integral_1^(1+z) u^(-3) * (gamma + wa*(u-1))^(-beta) du
    = integral_1^(1+z) u^(-3) * (wa*u + (gamma - wa))^(-beta) du
    = (wa)^(-beta) * integral_1^(1+z) u^(-3) * (u + (gamma-wa)/wa)^(-beta) du
    
    This is a hypergeometric-type integral. For numerical robustness,
    we use a high-order Gauss-Legendre quadrature (64 points) which is
    extremely fast and accurate for smooth integrands.
    """
    # Use 64-point Gauss-Legendre on [0, z] — very fast and accurate
    # Pre-compute nodes/weights once (cached)
    if not hasattr(_exact_integral_closed_form, '_gl64'):
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
        _exact_integral_closed_form._gl64 = (gl_nodes, gl_weights)
    gl_nodes, gl_weights = _exact_integral_closed_form._gl64
    
    # Map from [-1,1] to [0, z]
    t_nodes = 0.5 * z * gl_nodes + 0.5 * z
    w_scaled = 0.5 * z * gl_weights
    
    a_nodes = 1.0 / (1.0 + t_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    return np.sum(w_scaled / E_vals)


def _quadrature_value(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    return z * np.sum(weights / E_vals)


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    def _time_left():
        return deadline - time.time()
    
    # Representative CPL models covering a broad parameter space
    # The validation will use disjoint cosmologies, so we need good coverage
    models = [
        (0.0, 0.0),      # Lambda-CDM
        (-1.0, 0.0),     # Cosmological constant
        (-0.5, -0.5),    # Quintessence
        (-1.0, 0.5),     # Phantom
        (-0.8, -0.5),    # Quintessence variant
        (-1.2, 0.3),     # Strong phantom
        (-0.3, -0.3),    # Mild quintessence
        (-1.0, -0.5),    # Mixed
        (-0.7, 0.5),     # Another variant
        (-1.5, 0.8),     # Extreme phantom
        (-0.5, 0.5),     # Quintessence with positive wa
        (-1.0, 1.0),     # Strong phantom wa
    ]
    
    # Dense redshift grid for minimax
    z_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8,
                       1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    
    # Pre-compute exact integrals for all (model, z) pairs
    n_models = len(models)
    n_z = len(z_grid)
    exact_vals = np.zeros((n_models, n_z))
    for i, (w0, wa) in enumerate(models):
        for j, z in enumerate(z_grid):
            exact_vals[i, j] = _exact_integral_closed_form(z, w0, wa)
    
    # Initial guess: Gauss-Legendre on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    nodes0 = np.clip(nodes0, 1e-8, 1.0 - 1e-8)
    nodes0 = np.sort(nodes0)
    
    # Parameterization: nodes via logit, weights via log
    # nodes_i = 1/(1+exp(-x_i)) to ensure (0,1)
    # weights_i = exp(y_i) / sum(exp(y_j)) to ensure positive, sum=1
    
    def _params_to_nw(params):
        """Convert 16 params to (nodes, weights)."""
        node_params = params[:8]
        weight_params = params[8:]
        # Sigmoid for nodes: ensures (0,1)
        nodes = 1.0 / (1.0 + np.exp(-node_params))
        # Clip to avoid numerical issues
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        # Sort nodes to ensure increasing
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        # Enforce strict increase with minimum gap
        for k in range(7):
            if nodes[k+1] - nodes[k] < 1e-8:
                nodes[k+1] = nodes[k] + 1e-8
        # Re-clip
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Softmax for weights
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        # Reorder weights to match sorted nodes
        weights = weights[sort_idx]
        
        return nodes, weights
    
    def _objective(params):
        if _time_left() < 0.5:
            return 1e10
        
        nodes, weights = _params_to_nw(params)
        
        # Compute quadrature values for all (model, z) pairs
        # Vectorized: nodes shape (8,), z_grid shape (n_z,)
        # z_nodes[i,j] = z_grid[j] * nodes[i]  -> shape (8, n_z)
        z_nodes = nodes[:, None] * z_grid[None, :]  # (8, n_z)
        a_nodes = 1.0 / (1.0 + z_nodes)  # (8, n_z)
        
        # E values: need to compute for each model
        # E(a, w0, wa) = a^(-3) * (1+w0+wa*(1/a-1))^(wa/(1+w0))
        # For vectorization over models:
        max_err = 0.0
        for i, (w0, wa) in enumerate(models):
            # a_nodes: (8, n_z)
            z_vals = 1.0 / a_nodes - 1.0  # (8, n_z)
            base = 1.0 + w0 + wa * z_vals  # (8, n_z)
            base = np.maximum(base, 1e-15)
            exponent = wa / (1.0 + w0)
            E_vals = a_nodes**(-3.0) * base**exponent  # (8, n_z)
            
            # Quadrature: z * sum_i w_i / E_i for each z
            # inv_E: (8, n_z), weights: (8,)
            inv_E = 1.0 / E_vals  # (8, n_z)
            quad_vals = z_grid[None, :] * np.sum(weights[:, None] * inv_E, axis=0)  # (n_z,)
            
            exact_row = exact_vals[i, :]  # (n_z,)
            
            # Relative error
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs((quad_vals - exact_row) / exact_row)
            rel_err = np.nan_to_num(rel_err, nan=1e10, posinf=1e10, neginf=1e10)
            
            err = np.max(rel_err)
            if err > max_err:
                max_err = err
        
        return max_err * 100.0  # In percent
    
    # Initial params
    # Invert sigmoid: x = log(n/(1-n))
    node_params0 = np.log(nodes0 / (1.0 - nodes0))
    weight_params0 = np.log(weights0)
    params0 = np.concatenate([node_params0, weight_params0])
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # Phase 1: Differential evolution for global exploration
    if _time_left() > 30.0:
        try:
            bounds = [(-5, 5)] * 8 + [(-5, 5)] * 8
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if _time_left() > 10.0:
        try:
            nm_result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_params = nm_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional local refinement from slightly perturbed starts
    if _time_left() > 5.0:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if _time_left() < 2.0:
                break
            perturbed = best_params + rng.normal(0, 0.1, size=16)
            try:
                nm2 = minimize(
                    _objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if nm2.fun < best_score:
                    best_score = nm2.fun
                    best_params = nm2.x.copy()
            except Exception:
                pass
    
    # Final extraction
    nodes_opt, weights_opt = _params_to_nw(best_params)
    
    # Verify constraints
    if not (np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and np.all(np.diff(nodes_opt) > 0)):
        # Fallback to Gauss-Legendre
        nodes_opt = nodes0
        weights_opt = weights0
    
    # Ensure weights sum to 1
    weights_opt = weights_opt / np.sum(weights_opt)
    
    return {"nodes": nodes_opt, "weights": weights_opt}
# EVOLVE-BLOCK-END
