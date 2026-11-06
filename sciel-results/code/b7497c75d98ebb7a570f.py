"""Improved program: start from Gauss-Legendre on [0,1] and optimize."""
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
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


def _exact_comoving(z, w0, wa, n_quad=2000):
    """Exact comoving distance ∫₀ᶻ dz'/E(z') via high-order Gauss-Legendre."""
    if z <= 0:
        return 0.0
    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    z_vals = 0.5 * z * (xs + 1.0)
    E_vals = _cpl_E(z_vals, w0, wa)
    return 0.5 * z * np.sum(ws / E_vals)


def _quad_comoving(z, w0, wa, nodes, weights):
    """Quadrature comoving distance: z * Σ wᵢ/E(z·uᵢ)."""
    if z <= 0:
        return 0.0
    z_vals = z * nodes
    E_vals = _cpl_E(z_vals, w0, wa)
    return z * np.sum(weights / E_vals)


def _worst_error(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=1500):
    """Compute worst relative error over ensemble (vectorized over z)."""
    worst = 0.0
    n_z = len(z_vals)
    for w0 in w0_vals:
        for wa in wa_vals:
            # Vectorize over z
            z_arr = np.array(z_vals)
            # Exact: loop over z for leggauss (can't easily vectorize leggauss)
            exact_arr = np.zeros(n_z)
            for i, z in enumerate(z_vals):
                exact_arr[i] = _exact_comoving(z, w0, wa, n_quad)
            # Quadrature: vectorize
            # z_arr[:,None] * nodes[None,:] gives (n_z, 8)
            z_nodes = np.outer(z_arr, nodes)
            E_mat = _cpl_E(z_nodes, w0, wa)  # (n_z, 8)
            quad_arr = z_arr * np.sum(weights[None, :] / E_mat, axis=1)
            # Relative error
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(quad_arr - exact_arr) / np.maximum(np.abs(exact_arr), 1e-30)
            worst = max(worst, float(np.max(rel_err)))
    return worst


def _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize nodes and weights."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    if not have_scipy:
        return best_nodes, best_weights
    
    def _params_to_nw(params):
        x = params[:n]
        y = params[n:]
        # Nodes: use cumulative exp to ensure (0,1) increasing
        ex = np.exp(x - np.max(x))
        cumsum = np.cumsum(ex)
        nodes_p = cumsum / cumsum[-1]
        eps = 1e-12
        nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
        for i in range(1, n):
            if nodes_p[i] <= nodes_p[i-1]:
                nodes_p[i] = nodes_p[i-1] + eps
        nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
        # Weights: exp parameterization
        ey = np.exp(y - np.max(y))
        weights_p = ey / ey.sum()
        return nodes_p, weights_p
    
    def _objective(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _worst_error(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
    
    # Initial params
    x_init = np.log(best_nodes)
    y_init = np.log(best_weights)
    p0 = np.concatenate([x_init, y_init])
    
    # Nelder-Mead
    t_start = time.time()
    res = minimize(
        _objective, p0,
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True,
                 'maxfev': 200}
    )
    if _deadline_ok():
        np_, wp_ = _params_to_nw(res.x)
        err_ = _worst_error(np_, wp_, w0_vals, wa_vals, z_vals)
        if err_ < best_err:
            best_nodes, best_weights, best_err = np_, wp_, err_
    
    # L-BFGS-B if time permits
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.6:
        p_cur = res.x if res.success else p0
        res2 = minimize(
            _objective, p_cur,
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-14, 'maxfun': 150}
        )
        if _deadline_ok():
            np2, wp2 = _params_to_nw(res2.x)
            err2 = _worst_error(np2, wp2, w0_vals, wa_vals, z_vals)
            if err2 < best_err:
                best_nodes, best_weights, best_err = np2, wp2, err2
    
    return best_nodes, best_weights


def construct():
    n = 8
    # Start from Gauss-Legendre nodes/weights on [0,1]
    # These are the optimal nodes/weights for integrating smooth functions on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    # Map from [-1,1] to [0,1]: u = (x+1)/2, weight scaled by 1/2
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    # Ensure strictly in (0,1)
    nodes_init = np.clip(nodes_init, 1e-15, 1.0 - 1e-15)
    for i in range(1, n):
        if nodes_init[i] <= nodes_init[i-1]:
            nodes_init[i] = nodes_init[i-1] + 1e-15
    nodes_init[-1] = min(nodes_init[-1], 1.0 - 1e-15)
    weights_init = weights_init / weights_init.sum()
    
    # Broad ensemble for optimization
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    time_budget = 80.0
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
