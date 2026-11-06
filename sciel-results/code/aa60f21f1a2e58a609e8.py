"""Improved program: Alternating node/weight optimization with differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


# Cache for exact integration to avoid recomputing Gauss-Legendre nodes
_GL_CACHE = {}

def _get_gl(n_quad):
    if n_quad not in _GL_CACHE:
        _GL_CACHE[n_quad] = np.polynomial.legendre.leggauss(n_quad)
    return _GL_CACHE[n_quad]


def _exact_comoving_fast(z_arr, w0, wa, n_quad=128):
    """Exact comoving distance via Gauss-Legendre with cached nodes."""
    xs, ws = _get_gl(n_quad)
    z_arr = np.asarray(z_arr)
    if z_arr.ndim == 0:
        z_arr = z_arr.reshape(1)
    z_points = 0.5 * z_arr[:, None] * (xs[None, :] + 1.0)
    E_vals = _cpl_E(z_points, w0, wa)
    integral_vals = 0.5 * z_arr * np.sum(ws[None, :] / E_vals, axis=1)
    return integral_vals


def _quad_comoving(z_arr, w0, wa, nodes, weights):
    """Quadrature comoving distance: z * Σ wᵢ/E(z·uᵢ)."""
    z_arr = np.asarray(z_arr)
    if z_arr.ndim == 0:
        z_arr = z_arr.reshape(1)
    z_nodes = z_arr[:, None] * nodes[None, :]
    E_vals = _cpl_E(z_nodes, w0, wa)
    quad_arr = z_arr * np.sum(weights[None, :] / E_vals, axis=1)
    return quad_arr


def _worst_error_exact(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute exact worst relative error."""
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


def _validate_nodes(nodes):
    """Ensure nodes are strictly increasing in (0,1)."""
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes[-1] = min(nodes[-1], 1.0 - 1e-10)
    return nodes


def _validate_weights(weights):
    """Ensure weights are positive and sum to 1."""
    weights = np.maximum(weights, 1e-15)
    weights = weights / weights.sum()
    return weights


def _optimize_nodes_fixed_weights(weights, w0_vals, wa_vals, z_vals, time_budget, rng):
    """Differential evolution to optimize nodes with fixed weights."""
    from scipy.optimize import differential_evolution
    
    n = 8
    z_arr = np.array(z_vals)
    
    def _node_objective(node_params):
        if not _deadline_ok():
            return 1e10
        # node_params are in (0,1), need to enforce ordering
        # Use sorting trick: sort the params to get ordered nodes
        nodes_sorted = np.sort(node_params)
        nodes_sorted = _validate_nodes(nodes_sorted)
        # Compute worst error
        worst = 0.0
        for w0 in w0_vals:
            for wa in wa_vals:
                exact_arr = _exact_comoving_fast(z_arr, w0, wa)
                quad_arr = _quad_comoving(z_arr, w0, wa, nodes_sorted, weights)
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
                current_max = np.max(rel_err)
                if current_max > worst:
                    worst = float(current_max)
        return worst
    
    # Bounds: each node in (0.01, 0.99)
    bounds = [(0.01, 0.99)] * n
    
    t_start = time.time()
    max_iter = max(5, int(time_budget * 0.6 / 0.5))  # ~0.5s per iteration estimate
    
    try:
        result = differential_evolution(
            _node_objective, bounds,
            seed=rng.randint(10000),
            maxiter=max_iter,
            tol=1e-8,
            popsize=10,
            polish=True,
            init='sobol',
            update_interval=5
        )
        if result.success and result.fun < 1e9:
            return _validate_nodes(np.sort(result.x))
    except Exception:
        pass
    
    # Fallback: return current best from local search
    return None


def _optimize_weights_fixed_nodes(nodes, w0_vals, wa_vals, z_vals, time_budget, rng):
    """Differential evolution to optimize weights with fixed nodes."""
    from scipy.optimize import differential_evolution
    
    n = 8
    z_arr = np.array(z_vals)
    
    def _weight_objective(weight_params):
        if not _deadline_ok():
            return 1e10
        # weight_params are positive, normalize
        weights_raw = np.exp(weight_params - np.max(weight_params))
        weights_raw = weights_raw / weights_raw.sum()
        # Compute worst error
        worst = 0.0
        for w0 in w0_vals:
            for wa in wa_vals:
                exact_arr = _exact_comoving_fast(z_arr, w0, wa)
                quad_arr = _quad_comoving(z_arr, w0, wa, nodes, weights_raw)
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
                current_max = np.max(rel_err)
                if current_max > worst:
                    worst = float(current_max)
        return worst
    
    # Bounds for log-weights: -10 to 10
    bounds = [(-10, 10)] * n
    
    t_start = time.time()
    max_iter = max(5, int(time_budget * 0.6 / 0.5))
    
    try:
        result = differential_evolution(
            _weight_objective, bounds,
            seed=rng.randint(10000),
            maxiter=max_iter,
            tol=1e-8,
            popsize=10,
            polish=True,
            init='sobol',
            update_interval=5
        )
        if result.success and result.fun < 1e9:
            weights_raw = np.exp(result.x - np.max(result.x))
            weights_raw = weights_raw / weights_raw.sum()
            return _validate_weights(weights_raw)
    except Exception:
        pass
    
    return None


def _local_refine_nodes(weights, w0_vals, wa_vals, z_vals, nodes_init, time_budget, rng):
    """Local refinement of nodes using Nelder-Mead."""
    from scipy.optimize import minimize
    
    n = 8
    z_arr = np.array(z_vals)
    
    def _obj(params):
        if not _deadline_ok():
            return 1e10
        nodes_p = _validate_nodes(np.sort(params))
        worst = 0.0
        for w0 in w0_vals:
            for wa in wa_vals:
                exact_arr = _exact_comoving_fast(z_arr, w0, wa)
                quad_arr = _quad_comoving(z_arr, w0, wa, nodes_p, weights)
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
                current_max = np.max(rel_err)
                if current_max > worst:
                    worst = float(current_max)
        return worst
    
    p0 = np.sort(nodes_init)
    
    try:
        result = minimize(_obj, p0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < 1e9:
            return _validate_nodes(np.sort(result.x))
    except Exception:
        pass
    
    return None


def _local_refine_weights(nodes, w0_vals, wa_vals, z_vals, weights_init, time_budget, rng):
    """Local refinement of weights using Nelder-Mead on log-weights."""
    from scipy.optimize import minimize
    
    n = 8
    z_arr = np.array(z_vals)
    
    def _obj(params):
        if not _deadline_ok():
            return 1e10
        weights_raw = np.exp(params - np.max(params))
        weights_raw = weights_raw / weights_raw.sum()
        worst = 0.0
        for w0 in w0_vals:
            for wa in wa_vals:
                exact_arr = _exact_comoving_fast(z_arr, w0, wa)
                quad_arr = _quad_comoving(z_arr, w0, wa, nodes, weights_raw)
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
                current_max = np.max(rel_err)
                if current_max > worst:
                    worst = float(current_max)
        return worst
    
    p0 = np.log(weights_init)
    
    try:
        result = minimize(_obj, p0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < 1e9:
            weights_raw = np.exp(result.x - np.max(result.x))
            weights_raw = weights_raw / weights_raw.sum()
            return _validate_weights(weights_raw)
    except Exception:
        pass
    
    return None


def _alternating_optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Alternating optimization: fix weights, optimize nodes; fix nodes, optimize weights."""
    n = 8
    rng = np.random.RandomState(42)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error_exact(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    t_start = time.time()
    
    # Alternate between node and weight optimization
    for iteration in range(20):
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.85:
            break
        
        # Optimize nodes (fix weights)
        if (time.time() - t_start) < time_budget * 0.8:
            node_time = min(5.0, time_budget * 0.3)
            new_nodes = _optimize_nodes_fixed_weights(
                best_weights, w0_vals, wa_vals, z_vals, node_time, rng)
            if new_nodes is not None:
                err_test = _worst_error_exact(new_nodes, best_weights, w0_vals, wa_vals, z_vals)
                if err_test < best_err:
                    best_nodes = new_nodes
                    best_err = err_test
                    print(f"Iter {iteration}: nodes improved, err={best_err:.6e}")
        
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.85:
            break
        
        # Optimize weights (fix nodes)
        if (time.time() - t_start) < time_budget * 0.8:
            weight_time = min(5.0, time_budget * 0.3)
            new_weights = _optimize_weights_fixed_nodes(
                best_nodes, w0_vals, wa_vals, z_vals, weight_time, rng)
            if new_weights is not None:
                err_test = _worst_error_exact(best_nodes, new_weights, w0_vals, wa_vals, z_vals)
                if err_test < best_err:
                    best_weights = new_weights
                    best_err = err_test
                    print(f"Iter {iteration}: weights improved, err={best_err:.6e}")
        
        # Local refinement
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.85:
            break
        
        # Refine nodes locally
        if (time.time() - t_start) < time_budget * 0.8:
            local_nodes = _local_refine_nodes(
                best_weights, w0_vals, wa_vals, z_vals, best_nodes, 3.0, rng)
            if local_nodes is not None:
                err_test = _worst_error_exact(local_nodes, best_weights, w0_vals, wa_vals, z_vals)
                if err_test < best_err:
                    best_nodes = local_nodes
                    best_err = err_test
        
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.85:
            break
        
        # Refine weights locally
        if (time.time() - t_start) < time_budget * 0.8:
            local_weights = _local_refine_weights(
                best_nodes, w0_vals, wa_vals, z_vals, best_weights, 3.0, rng)
            if local_weights is not None:
                err_test = _worst_error_exact(best_nodes, local_weights, w0_vals, wa_vals, z_vals)
                if err_test < best_err:
                    best_weights = local_weights
                    best_err = err_test
    
    return best_nodes, best_weights


def construct():
    n = 8
    # Start from Gauss-Legendre nodes/weights on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    
    # Ensure strictly in (0,1)
    nodes_init = _validate_nodes(nodes_init)
    weights_init = _validate_weights(weights_init)
    
    # Ensemble for optimization
    # Broad range of w0, wa
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # z values: weighted toward high z where errors are larger
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    
    time_budget = 78.0
    t_start = time.time()
    
    try:
        nodes, weights = _alternating_optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
    except Exception:
        nodes, weights = nodes_init, weights_init
    
    # Final cleanup
    nodes = _validate_nodes(nodes)
    weights = _validate_weights(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
