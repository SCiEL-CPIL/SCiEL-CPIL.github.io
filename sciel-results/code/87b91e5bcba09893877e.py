"""Improved quadrature rule discovery for luminosity distance."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 80.0

def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat universe with matter + CPL dark energy."""
    Om = 0.3
    Ode = 1.0 - Om
    term_m = Om / (a ** 3)
    exponent = 3.0 * wa * (a - 1.0) + 3.0 * (w0 + wa) * (1.0 - a)
    term_de = Ode * np.exp(exponent)
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))

def _integrand_at(a, z, w0, wa):
    """The integrand f(a) = 1/E(a) evaluated at scale factor a for given z."""
    return 1.0 / _E_of_a(a, w0, wa)

def _true_value(z, w0, wa, n_pts=500):
    """High-accuracy reference value using dense quadrature."""
    x, w = np.polynomial.legendre.leggauss(n_pts)
    a_pts = 1.0 - z * 0.5 * (x + 1.0)
    # a goes from 1 (x=-1) to 1/(1+z) (x=1)
    # Integral is ∫ from a=1/(1+z) to a=1 of da/E(a)
    # Change variable: x from -1 to 1, a = 1 - z/2*(x+1)
    # da/dx = -z/2
    # So ∫ da/E(a) from a_low to a_high = ∫_{x=1}^{x=-1} (-z/2) / E(a(x)) dx
    # = ∫_{-1}^{1} z/2 / E(a(x)) dx
    da_dx = z / 2.0
    vals = da_dx * _integrand_at(a_pts, z, w0, wa) * w
    integral = np.sum(vals)
    return (1.0 + z) * integral

def _quad_value(nodes, weights, z, w0, wa):
    """Compute quadrature approximation."""
    a_i = 1.0 / (1.0 + z * nodes)
    E_i = _E_of_a(a_i, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_i)

def _rel_err(nodes, weights, z, w0, wa):
    """Relative error for a single (z, w0, wa)."""
    qv = _quad_value(nodes, weights, z, w0, wa)
    rv = _true_value(z, w0, wa)
    if abs(rv) < 1e-15:
        return 0.0
    return abs(qv - rv) / abs(rv)

def _worst_err(nodes, weights, z_vals, models):
    """Worst relative error over z and models."""
    max_e = 0.0
    for w0, wa in models:
        for z in z_vals:
            e = _rel_err(nodes, weights, z, w0, wa)
            if e > max_e:
                max_e = e
    return max_e

def _penalized_objective(params):
    """Objective: worst error + penalty for constraint violations."""
    if time.time() > _DEADLINE - 5.0:
        return 1e10
    nodes = params[:8]
    log_w = params[8:]
    weights = np.exp(log_w)
    weights = weights / np.sum(weights)
    
    # Check node constraints
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    
    z_test = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0])
    models = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5)]
    
    return _worst_err(nodes, weights, z_test, models)

def _optimize_from(nodes, weights, max_iter=200):
    """Optimize starting from given nodes and weights."""
    log_w = np.log(np.maximum(weights, 1e-30))
    params0 = np.concatenate([nodes, log_w])
    
    try:
        res = minimize(_penalized_objective, params0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 
                               'fatol': 1e-12, 'adaptive': True,
                               'initial_simplex': 1e-4})
        if res.fun < 1e9:
            new_nodes = res.x[:8]
            new_weights = np.exp(res.x[8:])
            new_weights = new_weights / np.sum(new_weights)
            return new_nodes, new_weights
    except Exception:
        pass
    return nodes, weights

def construct():
    t_start = time.time()
    
    # Multiple diverse starting points
    starting_points = []
    
    # 1. Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    starting_points.append((0.5 * (gl_nodes + 1.0), 0.5 * gl_weights))
    
    # 2. Chebyshev nodes
    k = np.arange(8)
    chev_nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8))
    chev_weights = np.ones(8) / 8.0
    starting_points.append((chev_nodes, chev_weights))
    
    # 3. Equally spaced
    eq_nodes = (np.arange(8) + 0.5) / 8.0
    eq_weights = np.ones(8) / 8.0
    starting_points.append((eq_nodes, eq_weights))
    
    # 4. Gauss-Jacobi-like (biased toward 0)
    gj_nodes = np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99])
    gj_weights = np.array([0.1, 0.15, 0.15, 0.15, 0.15, 0.12, 0.12, 0.06])
    gj_weights = gj_weights / np.sum(gj_weights)
    starting_points.append((gj_nodes, gj_weights))
    
    # 5. Another variant
    alt_nodes = np.array([0.03, 0.1, 0.25, 0.45, 0.65, 0.8, 0.92, 0.98])
    alt_weights = np.array([0.08, 0.12, 0.15, 0.15, 0.15, 0.13, 0.14, 0.08])
    alt_weights = alt_weights / np.sum(alt_weights)
    starting_points.append((alt_nodes, alt_weights))
    
    # Evaluate all starting points
    z_eval = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0])
    models_eval = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5), (0.0, -0.5)]
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    for nodes, weights in starting_points:
        err = _worst_err(nodes, weights, z_eval, models_eval)
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    # Optimize from each starting point
    for nodes, weights in starting_points:
        if time.time() > _DEADLINE - 30.0:
            break
        try:
            refined_nodes, refined_weights = _optimize_from(nodes.copy(), weights.copy(), max_iter=300)
            if (np.all(np.diff(refined_nodes) > 0) and
                np.all(refined_nodes > 1e-8) and
                np.all(refined_nodes < 1.0 - 1e-8) and
                np.all(refined_weights > 0)):
                refined_weights = refined_weights / np.sum(refined_weights)
                err = _worst_err(refined_nodes, refined_weights, z_eval, models_eval)
                if err < best_err:
                    best_err = err
                    best_nodes = refined_nodes
                    best_weights = refined_weights
        except Exception:
            pass
    
    # Second round of refinement from best
    if time.time() < _DEADLINE - 20.0 and best_nodes is not None:
        try:
            refined_nodes, refined_weights = _optimize_from(best_nodes.copy(), best_weights.copy(), max_iter=200)
            if (np.all(np.diff(refined_nodes) > 0) and
                np.all(refined_nodes > 1e-8) and
                np.all(refined_nodes < 1.0 - 1e-8) and
                np.all(refined_weights > 0)):
                refined_weights = refined_weights / np.sum(refined_weights)
                err = _worst_err(refined_nodes, refined_weights, z_eval, models_eval)
                if err < best_err:
                    best_err = err
                    best_nodes = refined_nodes
                    best_weights = refined_weights
        except Exception:
            pass
    
    # Final validation with broader set
    if time.time() < _DEADLINE - 10.0 and best_nodes is not None:
        z_final = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
        models_final = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5), 
                       (0.0, -0.5), (0.5, 0.0), (-1.0, 0.5)]
        final_err = _worst_err(best_nodes, best_weights, z_final, models_final)
        
        # If significantly worse, try a fallback
        if final_err > best_err * 1.5:
            # Fall back to best starting point
            for nodes, weights in starting_points:
                err = _worst_err(nodes, weights, z_final, models_final)
                if err < final_err:
                    final_err = err
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
    
    # Ensure final constraints
    if best_nodes is None:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        best_nodes = 0.5 * (gl_nodes + 1.0)
        best_weights = 0.5 * gl_weights
    
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure strictly increasing and in (0,1)
    for i in range(7):
        if best_nodes[i+1] <= best_nodes[i]:
            best_nodes[i+1] = best_nodes[i] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
