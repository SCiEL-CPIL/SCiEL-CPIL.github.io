"""Improved program: Fast vectorized optimization of 8-node universal quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry.
    E(z)^2 = (1+z)^{3(1+w0+wa)} * (1 + wa*z)^{-3*wa}
    """
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    # (1+z)^{3(1+w0+wa)} * (1+wa*z)^{-3*wa}
    # Use np.power for vectorization
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


def _exact_comoving_fast(z_arr, w0, wa, n_quad=256):
    """Exact comoving distance for an array of z values via high-order Gauss-Legendre.
    Returns array of same shape as z_arr.
    """
    # Pre-compute nodes and weights for the reference integral
    # We map [-1, 1] to [0, z] for each z
    # To vectorize, we can compute the integral for a fixed set of z values
    # Integral I(z) = 0.5 * z * sum_{j=1}^{n_quad} w_j / E(0.5 * z * (x_j + 1))
    
    # z_arr shape: (N,)
    # We need to compute E for (N, n_quad) points
    # z_points = 0.5 * z_arr[:, None] * (xs[None, :] + 1.0)
    
    # Since n_quad is fixed, we can compute xs, ws once outside if possible, 
    # but here we assume it's called multiple times with different w0, wa.
    # However, xs and ws depend only on n_quad.
    
    # To save time, we could cache xs, ws. But for simplicity and memory, 
    # we compute them. 256 points is fast.
    
    # Optimization: If z_arr is small, this is very fast.
    # If z_arr is large, it might be memory heavy but 256 is manageable.
    
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    # Using np.polynomial.legendre.leggauss
    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    
    # Map to [0, z]
    # z_points[i, j] = 0.5 * z_arr[i] * (xs[j] + 1.0)
    # This creates an (N, n_quad) array
    z_points = 0.5 * z_arr[:, None] * (xs[None, :] + 1.0)
    
    # Compute E(z_points)
    # w0, wa are scalars
    E_vals = _cpl_E(z_points, w0, wa)
    
    # Integrate: sum(ws / E_vals, axis=1) * 0.5 * z_arr
    # Note: ws are for [-1, 1], so the factor 0.5*z is the Jacobian
    integral_vals = 0.5 * z_arr * np.sum(ws[None, :] / E_vals, axis=1)
    
    return integral_vals


def _quad_comoving(z_arr, w0, wa, nodes, weights):
    """Quadrature comoving distance: z * Σ wᵢ/E(z·uᵢ)."""
    # z_arr shape: (N,)
    # nodes shape: (8,)
    # z_nodes shape: (N, 8)
    z_nodes = z_arr[:, None] * nodes[None, :]
    E_vals = _cpl_E(z_nodes, w0, wa)
    # Sum over nodes
    quad_arr = z_arr * np.sum(weights[None, :] / E_vals, axis=1)
    return quad_arr


def _worst_error_smooth(nodes, weights, w0_vals, wa_vals, z_vals, temperature=0.01):
    """Compute smooth approximation of worst relative error.
    Uses log-sum-exp to approximate max.
    """
    n_z = len(z_vals)
    z_arr = np.array(z_vals)
    
    errors = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            # Exact
            exact_arr = _exact_comoving_fast(z_arr, w0, wa)
            # Quadrature
            quad_arr = _quad_comoving(z_arr, w0, wa, nodes, weights)
            
            # Relative error
            # Avoid division by zero
            rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            errors.append(rel_err)
            
    # Concatenate all errors
    all_errors = np.concatenate(errors)
    
    # Smooth max: T * log(sum(exp(err/T)))
    # If T is small, this approximates max
    max_err = np.max(all_errors)
    # To avoid overflow in exp, subtract max
    # log(sum(exp((err - max)/T))) + max
    # But we want the value itself.
    # Actually, we just need to minimize this.
    # Let's use a simpler smooth max if temperature is small enough.
    
    # Using log-sum-exp trick
    shifted = all_errors - max_err
    log_sum_exp = np.log(np.sum(np.exp(shifted / temperature))) + max_err / temperature
    
    # The smooth max is T * log_sum_exp
    smooth_max = temperature * log_sum_exp
    
    return float(smooth_max)


def _worst_error_exact(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute exact worst relative error (for final validation)."""
    n_z = len(z_vals)
    z_arr = np.array(z_vals)
    worst = 0.0
    
    for w0 in w0_vals:
        for wa in wa_vals:
            exact_arr = _exact_comoving_fast(z_arr, w0, wa)
            quad_arr = _quad_comoving(z_arr, w0, wa, nodes, weights)
            
            rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            current_max = np.max(rel_err)
            if current_max > worst:
                worst = float(current_max)
                
    return worst


def _params_to_nw(params):
    """Convert parameters to nodes and weights.
    nodes: log-space cumulative sum to ensure (0,1) increasing.
    weights: log-space to ensure positive, normalized.
    """
    n = 8
    x = params[:n]
    y = params[n:]
    
    # Nodes
    # Use softplus or exp to ensure positive differences
    # Let d_i = exp(x_i - max(x)). Then nodes are cumulative sums normalized.
    ex = np.exp(x - np.max(x))
    cumsum = np.cumsum(ex)
    nodes_p = cumsum / cumsum[-1]
    
    # Ensure strictly in (0, 1)
    eps = 1e-12
    nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
    for i in range(1, n):
        if nodes_p[i] <= nodes_p[i-1]:
            nodes_p[i] = nodes_p[i-1] + eps
    nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
    
    # Weights
    ey = np.exp(y - np.max(y))
    weights_p = ey / ey.sum()
    
    return nodes_p, weights_p


def _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize nodes and weights."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error_exact(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    if not have_scipy:
        return best_nodes, best_weights
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        # Use smooth error for optimization
        return _worst_error_smooth(nodes_p, weights_p, w0_vals, wa_vals, z_vals, temperature=0.05)
    
    # Initial params
    x_init = np.log(best_nodes)
    y_init = np.log(best_weights)
    p0 = np.concatenate([x_init, y_init])
    
    t_start = time.time()
    
    # Multi-start L-BFGS-B
    # We can try a few random perturbations
    rng = np.random.RandomState(42)
    
    for i in range(5):
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.8:
            break
            
        if i == 0:
            p_cur = p0
        else:
            # Perturb initial guess
            p_cur = p0 + rng.normal(0, 0.1, size=2*n)
            
        try:
            res = minimize(
                _objective, p_cur,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12, 'maxfun': 200}
            )
            
            if _deadline_ok():
                np_, wp_ = _params_to_nw(res.x)
                # Check exact error to ensure improvement
                err_ = _worst_error_exact(np_, wp_, w0_vals, wa_vals, z_vals)
                if err_ < best_err:
                    best_nodes, best_weights, best_err = np_, wp_, err_
        except Exception:
            continue
            
    return best_nodes, best_weights


def construct():
    n = 8
    # Start from Gauss-Legendre nodes/weights on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    
    # Ensure strictly in (0,1)
    nodes_init = np.clip(nodes_init, 1e-15, 1.0 - 1e-15)
    for i in range(1, n):
        if nodes_init[i] <= nodes_init[i-1]:
            nodes_init[i] = nodes_init[i-1] + 1e-15
    nodes_init[-1] = min(nodes_init[-1], 1.0 - 1e-15)
    weights_init = weights_init / weights_init.sum()
    
    # Ensemble for optimization
    # Broad range of w0, wa
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # z values: mix of small, medium, large
    z_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    time_budget = 75.0
    t_start = time.time()
    
    try:
        nodes, weights = _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
    except Exception:
        nodes, weights = nodes_init, weights_init
    
    # Final cleanup
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    nodes[-1] = min(nodes[-1], 1.0 - 1e-12)
    weights = np.maximum(weights, 1e-15)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
