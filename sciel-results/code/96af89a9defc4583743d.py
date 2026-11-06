"""Combined program: smooth max objective with multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


# Pre-compute Gauss-Legendre nodes for exact integration
_GX64, _GW64 = np.polynomial.legendre.leggauss(64)


def _exact_comoving_fast(z, w0, wa):
    """Exact comoving distance using pre-computed 64-point Gauss-Legendre."""
    if z <= 0:
        return 0.0
    z_vals = 0.5 * z * (_GX64 + 1.0)
    E_vals = _cpl_E(z_vals, w0, wa)
    return 0.5 * z * np.sum(_GW64 / E_vals)


def _quad_comoving(z, w0, wa, nodes, weights):
    """Quadrature comoving distance."""
    if z <= 0:
        return 0.0
    z_vals = z * nodes
    E_vals = _cpl_E(z_vals, w0, wa)
    return z * np.sum(weights / E_vals)


def _worst_error_smooth(nodes, weights, w0_vals, wa_vals, z_vals, beta=50.0):
    """Compute smooth max (log-sum-exp) of relative errors."""
    errors = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                exact = _exact_comoving_fast(z, w0, wa)
                quad = _quad_comoving(z, w0, wa, nodes, weights)
                rel_err = abs(quad - exact) / max(abs(exact), 1e-30)
                errors.append(rel_err)
    errors = np.array(errors)
    # Log-sum-exp for smooth max
    m = np.max(errors)
    if m < 1e-15:
        return m
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (errors - m)))) / beta
    return log_sum_exp


def _worst_error_exact(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute exact worst relative error."""
    worst = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                exact = _exact_comoving_fast(z, w0, wa)
                quad = _quad_comoving(z, w0, wa, nodes, weights)
                rel_err = abs(quad - exact) / max(abs(exact), 1e-30)
                worst = max(worst, rel_err)
    return worst


def _params_to_nw(params):
    """Convert parameters to valid nodes and weights."""
    n = 8
    x = params[:n]
    y = params[n:]
    # Nodes: cumulative exp ensures positive, increasing, in (0,1)
    ex = np.exp(x - np.max(x))
    cumsum = np.cumsum(ex)
    nodes_p = cumsum / cumsum[-1]
    eps = 1e-12
    nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
    for i in range(1, n):
        if nodes_p[i] <= nodes_p[i-1]:
            nodes_p[i] = nodes_p[i-1] + eps
    nodes_p[-1] = min(nodes_p[-1], 1.0 - eps)
    # Weights: exp parameterization ensures positive, sum to 1
    ey = np.exp(y - np.max(y))
    weights_p = ey / ey.sum()
    return nodes_p, weights_p


def _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize nodes and weights with smooth max objective."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error_exact(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    def _objective_smooth(params):
        if not _deadline_ok():
            return 1e10
        nodes_p, weights_p = _params_to_nw(params)
        return _worst_error_smooth(nodes_p, weights_p, w0_vals, wa_vals, z_vals, beta=100.0)
    
    # Initial params
    x_init = np.log(np.maximum(best_nodes, 1e-15))
    y_init = np.log(np.maximum(best_weights, 1e-15))
    p0 = np.concatenate([x_init, y_init])
    
    t_start = time.time()
    rng = np.random.RandomState(123)
    
    # Multi-start with diverse initializations
    for restart in range(8):
        if not _deadline_ok() or (time.time() - t_start) > time_budget * 0.75:
            break
        
        if restart == 0:
            p_cur = p0.copy()
        elif restart == 1:
            # Perturb moderately
            p_cur = p0 + rng.normal(0, 0.05, size=2*n)
        elif restart == 2:
            # Perturb more
            p_cur = p0 + rng.normal(0, 0.15, size=2*n)
        elif restart == 3:
            # Random restart from GL nodes with jitter
            xs, ws = np.polynomial.legendre.leggauss(n)
            nodes_r = 0.5 * (xs + 1.0) + rng.normal(0, 0.02, size=n)
            weights_r = 0.5 * ws + rng.normal(0, 0.01, size=n)
            nodes_r = np.clip(nodes_r, 1e-6, 1.0 - 1e-6)
            weights_r = np.maximum(weights_r, 1e-6)
            weights_r = weights_r / weights_r.sum()
            x_r = np.log(np.maximum(nodes_r, 1e-15))
            y_r = np.log(np.maximum(weights_r, 1e-15))
            p_cur = np.concatenate([x_r, y_r])
        else:
            # Perturb best found so far
            x_b = np.log(np.maximum(best_nodes, 1e-15))
            y_b = np.log(np.maximum(best_weights, 1e-15))
            p_best = np.concatenate([x_b, y_b])
            p_cur = p_best + rng.normal(0, 0.03 + 0.01*restart, size=2*n)
        
        # Nelder-Mead for robustness
        res = minimize(
            _objective_smooth, p_cur,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                     'maxfev': 120}
        )
        
        if _deadline_ok():
            np_, wp_ = _params_to_nw(res.x)
            err_ = _worst_error_exact(np_, wp_, w0_vals, wa_vals, z_vals)
            if err_ < best_err:
                best_nodes, best_weights, best_err = np_, wp_, err_
                p0 = res.x.copy()
    
    # Final L-BFGS-B polish with smooth objective
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.85:
        x_b = np.log(np.maximum(best_nodes, 1e-15))
        y_b = np.log(np.maximum(best_weights, 1e-15))
        p_best = np.concatenate([x_b, y_b])
        
        res2 = minimize(
            _objective_smooth, p_best,
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-13, 'maxfun': 120}
        )
        if _deadline_ok():
            np2, wp2 = _params_to_nw(res2.x)
            err2 = _worst_error_exact(np2, wp2, w0_vals, wa_vals, z_vals)
            if err2 < best_err:
                best_nodes, best_weights, best_err = np2, wp2, err2
    
    # Very tight final polish if time remains
    if _deadline_ok() and (time.time() - t_start) < time_budget * 0.92:
        x_b = np.log(np.maximum(best_nodes, 1e-15))
        y_b = np.log(np.maximum(best_weights, 1e-15))
        p_best = np.concatenate([x_b, y_b])
        
        # Use even smoother objective for final polish
        def _objective_final(params):
            if not _deadline_ok():
                return 1e10
            nodes_p, weights_p = _params_to_nw(params)
            return _worst_error_smooth(nodes_p, weights_p, w0_vals, wa_vals, z_vals, beta=200.0)
        
        res3 = minimize(
            _objective_final, p_best,
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-14, 'maxfun': 60}
        )
        if _deadline_ok():
            np3, wp3 = _params_to_nw(res3.x)
            err3 = _worst_error_exact(np3, wp3, w0_vals, wa_vals, z_vals)
            if err3 < best_err:
                best_nodes, best_weights, best_err = np3, wp3, err3
    
    return best_nodes, best_weights


def construct():
    n = 8
    # Start from Gauss-Legendre nodes/weights on [0,1]
    xs, ws = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (xs + 1.0)
    weights_init = 0.5 * ws
    nodes_init = np.clip(nodes_init, 1e-15, 1.0 - 1e-15)
    for i in range(1, n):
        if nodes_init[i] <= nodes_init[i-1]:
            nodes_init[i] = nodes_init[i-1] + 1e-15
    nodes_init[-1] = min(nodes_init[-1], 1.0 - 1e-15)
    weights_init = weights_init / weights_init.sum()
    
    # Focused ensemble: cover the parameter space where errors are likely worst
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    z_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    time_budget = 80.0
    
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
