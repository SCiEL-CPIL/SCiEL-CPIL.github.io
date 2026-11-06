"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = 85.0  # seconds, safety margin before 90s kill
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(a, w0, wa):
    """E(a) = H(a)/H0 for CPL model with flat spatial geometry."""
    # E(z)^2 = (1+z)^2 * (1+z)^(-3(1+w0+wa)) * (1+wa*z)
    # In terms of a: z = 1/a - 1
    # E(a)^2 = a^(-2) * a^(3(1+w0+wa)) * (1 + wa*(1/a - 1))
    #        = a^(3(w0+wa)-2) * (1 + wa*(1-a)/a)
    #        = a^(3(w0+wa)-2) * ((a + wa*(1-a))/a)
    #        = a^(3(w0+wa)-3) * (a + wa - wa*a)
    #        = a^(3(w0+wa)-3) * (1 + (a-1)*wa)
    # Let's be more careful:
    # E(z)^2 = (1+z)^2 * (1+z)^{-3(1+w0+wa)} * (1 + wa*z)
    # (1+z)^2 * (1+z)^{-3(1+w0+wa)} = (1+z)^{2-3(1+w0+wa)} = (1+z)^{-1-3(w0+wa)}
    # So E(z)^2 = (1+z)^{-1-3(w0+wa)} * (1 + wa*z)
    # In terms of a: 1+z = 1/a, z = (1-a)/a
    # E(a)^2 = a^{1+3(w0+wa)} * (1 + wa*(1-a)/a)
    #        = a^{1+3(w0+wa)} * (1 + wa - wa*a)/a
    #        = a^{3(w0+wa)} * (1 + wa - wa*a)
    #        = a^{3(w0+wa)} * (1 + wa*(1-a))
    w0w = w0 + wa
    return np.sqrt(a**(3.0*w0w) * (1.0 + wa*(1.0 - a)))


def _exact_int(w0, wa, z, n_quad=2000):
    """High-accuracy adaptive-like integration for reference."""
    # Integrand: (1+z')*z' / E(a') where a' = 1/(1+z')
    # We integrate over z' from 0 to z
    if z <= 0:
        return 0.0
    # Use Gauss-Legendre for accuracy
    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1,1] to [0, z]
    z_vals = 0.5 * z * (xs + 1.0)
    a_vals = 1.0 / (1.0 + z_vals)
    E_vals = _cpl_E(a_vals, w0, wa)
    integrand = (1.0 + z_vals) * z_vals / E_vals
    # Jacobian: dz'/dx = z/2
    return 0.5 * z * np.sum(ws * integrand)


def _quad_int(w0, wa, z, nodes, weights):
    """Quadrature-based luminosity distance."""
    if z <= 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _cpl_E(a_vals, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)


def _rel_errors(w0, wa, z, nodes, weights, n_quad=2000):
    """Relative error between quadrature and exact integration."""
    exact = _exact_int(w0, wa, z, n_quad)
    quad = _quad_int(w0, wa, z, nodes, weights)
    if abs(exact) < 1e-15:
        return 0.0
    return abs(quad - exact) / abs(exact)


def _fit_ensemble(nodes, weights, w0_vals, wa_vals, z_vals, n_quad=1000):
    """Compute worst-case relative error over ensemble."""
    worst = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                err = _rel_errors(w0, wa, z, nodes, weights, n_quad)
                if err > worst:
                    worst = err
    return worst


def _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget):
    """Optimize nodes and weights to minimize worst-case error."""
    n = 8
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _fit_ensemble(best_nodes, best_weights, w0_vals, wa_vals, z_vals)
    
    # Use scipy if available for more robust optimization
    try:
        from scipy.optimize import minimize
        from scipy.optimize import minimize_scalar
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    if have_scipy:
        # Parameterize nodes with monotonicity constraint via cumulative sums of positive values
        # nodes[i] = sum_{j<=i} exp(x[j]) / sum_{k} exp(x[k])
        # weights: use exp parameterization to ensure positivity, normalize
        
        def _params_to_nodes_weights(params):
            x = params[:n]
            y = params[n:]
            # Nodes: softmax-like to ensure in (0,1) and increasing
            # Use cumulative sum of exp
            ex = np.exp(x - np.max(x))  # numerical stability
            cumsum = np.cumsum(ex)
            nodes_p = cumsum / cumsum[-1]
            # Ensure strictly in (0,1): nodes_p[0] > 0, nodes_p[-1] = 1
            # Add small epsilon
            eps = 1e-10
            nodes_p = np.clip(nodes_p, eps, 1.0 - eps)
            # Ensure strictly increasing
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
            nodes_p, weights_p = _params_to_nodes_weights(params)
            err = _fit_ensemble(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
            return err
        
        # Initial parameterization
        x_init = np.log(best_nodes)  # rough
        y_init = np.log(best_weights)
        
        # Try Nelder-Mead
        t_start = time.time()
        res = minimize(
            _objective, 
            np.concatenate([x_init, y_init]),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if _deadline_ok():
            nodes_p, weights_p = _params_to_nodes_weights(res.x)
            err_p = _fit_ensemble(nodes_p, weights_p, w0_vals, wa_vals, z_vals)
            if err_p < best_err:
                best_nodes = nodes_p
                best_weights = weights_p
                best_err = err_p
        
        # Try L-BFGS-B with more iterations if time permits
        if _deadline_ok() and (time.time() - t_start) < time_budget * 0.5:
            res2 = minimize(
                _objective,
                res.x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            if _deadline_ok():
                nodes_p2, weights_p2 = _params_to_nodes_weights(res2.x)
                err_p2 = _fit_ensemble(nodes_p2, weights_p2, w0_vals, wa_vals, z_vals)
                if err_p2 < best_err:
                    best_nodes = nodes_p2
                    best_weights = weights_p2
                    best_err = err_p2
    
    # Final polish: coordinate descent if time permits
    if _deadline_ok():
        for _ in range(3):
            if not _deadline_ok():
                break
            improved = False
            for i in range(n):
                if not _deadline_ok():
                    break
                # Try small perturbations of node i
                for delta in [1e-4, -1e-4, 1e-5, -1e-5]:
                    new_nodes = best_nodes.copy()
                    new_nodes[i] += delta
                    # Check monotonicity and bounds
                    valid = True
                    for j in range(n):
                        if new_nodes[j] <= 0 or new_nodes[j] >= 1:
                            valid = False
                            break
                    if valid:
                        for j in range(1, n):
                            if new_nodes[j] <= new_nodes[j-1]:
                                valid = False
                                break
                    if valid:
                        err_new = _fit_ensemble(new_nodes, best_weights, w0_vals, wa_vals, z_vals)
                        if err_new < best_err:
                            best_nodes = new_nodes
                            best_err = err_new
                            improved = True
            # Try small perturbations of weights
            for i in range(n):
                if not _deadline_ok():
                    break
                for delta in [1e-4, -1e-4, 1e-5, -1e-5]:
                    new_weights = best_weights.copy()
                    new_weights[i] += delta
                    if np.all(new_weights > 0):
                        new_weights = new_weights / new_weights.sum()
                        err_new = _fit_ensemble(best_nodes, new_weights, w0_vals, wa_vals, z_vals)
                        if err_new < best_err:
                            best_weights = new_weights
                            best_err = err_new
                            improved = True
            if not improved:
                break
    
    # Ensure weights sum to 1
    best_weights = best_weights / best_weights.sum()
    return best_nodes, best_weights


def construct():
    n = 8
    # Initial guess: composite midpoint
    nodes_init = (np.arange(n) + 0.5) / n
    weights_init = np.full(n, 1.0 / n)
    
    # Ensemble of CPL models and redshifts
    w0_vals = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_vals = [-0.5, -0.3, 0.0, 0.3, 0.5, 0.8]
    z_vals = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    time_budget = 80.0  # leave 10s margin
    t_start = time.time()
    
    try:
        nodes, weights = _optimize(nodes_init, weights_init, w0_vals, wa_vals, z_vals, time_budget)
    except Exception:
        nodes = nodes_init
        weights = weights_init
    
    # Final validation and cleanup
    # Ensure nodes are strictly in (0,1) and increasing
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-12
    nodes[-1] = min(nodes[-1], 1.0 - 1e-12)
    
    # Ensure weights are positive and sum to 1
    weights = np.maximum(weights, 1e-15)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
