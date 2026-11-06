"""
Universal 8-node positive quadrature for dimensionless luminosity distance.

Design:
  dL/(c/H0) = (1+z) * z * sum_i w_i / E(a_i),  a_i = 1/(1+z*x_i)

We design nodes x_i in (0,1) and positive weights w_i (sum=1) to minimize
the worst relative error over a family of CPL dark-energy models.

Strategy:
  1. Define a set of representative CPL models (w0, wa) covering the
     physically plausible parameter space.
  2. For each model, compute the true integral I(z) = (1+z)*∫_0^z z'/E(a(z')) dz'
     via high-accuracy adaptive quadrature (as the evaluator does).
  3. The quadrature approximates I(z) ≈ (1+z)*z*Σ w_i / E(a(z*x_i)).
  4. Minimize worst-case relative error over a grid of z values and models.
  5. Use a two-stage approach:
     - Stage 1: coarse search over node configurations (Gauss-Legendre,
       Chebyshev, log-spaced, power-law, random perturbations).
     - Stage 2: local refinement via Nelder-Mead or L-BFGS-B on nodes+weights,
       with positivity constraint on weights.
  6. Weights are solved as a least-squares problem given nodes, then
     projected to be positive.

Time budget: 90 s wall-clock. We track elapsed time and return best found.
"""

import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad
import sys

# Deadline guard
_START_TIME = time.time()
DEADLINE = 90.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _time_remaining():
    return DEADLINE - (time.time() - _START_TIME) - SAFETY_MARGIN


def _E_of_z(z, w0, wa):
    """E(a) = H(a)/H0 for CPL model. a = 1/(1+z)."""
    a = 1.0 / (1.0 + z)
    return np.sqrt((a ** (-3.0 * (1.0 + w0 + wa))) * (a ** (-3.0 * wa)))


def _E_of_z_vec(z_arr, w0, wa):
    """Vectorized E(z)."""
    a = 1.0 / (1.0 + z_arr)
    return np.sqrt((a ** (-3.0 * (1.0 + w0 + wa))) * (a ** (-3.0 * wa)))


def _true_integrand(z, w0, wa):
    """Integrand for the luminosity distance integral: z / E(a(z))."""
    if z < 1e-15:
        return 0.0
    return z / _E_of_z(z, w0, wa)


def _true_luminosity_distance(z, w0, wa):
    """True dL/(c/H0) = (1+z) * ∫_0^z z'/E(a(z')) dz'."""
    if z < 1e-15:
        return 0.0
    val, _ = quad(_true_integrand, 0.0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-10)
    return (1.0 + z) * val


def _approx_luminosity_distance(z, nodes, weights, w0, wa):
    """Quadrature approximation: (1+z)*z*sum_i w_i / E(a(z*x_i))."""
    if z < 1e-15:
        return 0.0
    z_nodes = z * nodes
    E_vals = _E_of_z_vec(z_nodes, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)


def _worst_rel_error(nodes, weights, models, z_grid):
    """Compute worst relative error (in percent) over models and z_grid."""
    worst = 0.0
    for (w0, wa) in models:
        for z in z_grid:
            true_val = _true_luminosity_distance(z, w0, wa)
            approx_val = _approx_luminosity_distance(z, nodes, weights, w0, wa)
            if true_val > 1e-15:
                rel_err = abs(approx_val - true_val) / true_val
                if rel_err > worst:
                    worst = rel_err
    return worst * 100.0  # in percent


def _solve_weights(nodes, models, z_grid):
    """
    Given fixed nodes, find positive weights summing to 1 that minimize
    worst-case relative error. We formulate as a least-squares problem
    on the integrand values, then project to positive simplex.
    
    Actually, we use a direct optimization: minimize max relative error
    subject to w_i > 0, sum(w) = 1.
    """
    n = len(nodes)
    
    # Precompute integrand values for all models and z
    # The quadrature is: approx(z) = (1+z)*z*sum_i w_i * f_i(z)
    # where f_i(z) = 1/E(a(z*x_i))
    # true(z) = (1+z)*∫_0^z z'/E(a(z')) dz'
    
    # For a given z, the relative error is:
    # |(1+z)*z*sum w_i f_i(z) - (1+z)*I(z)| / ((1+z)*I(z))
    # = |z*sum w_i f_i(z) - I(z)| / I(z)
    
    # Let's build a system. For each (model, z) pair, we have:
    # z * sum_i w_i * f_i(z, model) ≈ I(z, model)
    # => sum_i w_i * f_i(z, model) ≈ I(z, model) / z
    
    n_samples = 0
    for (w0, wa) in models:
        for z in z_grid:
            if z > 1e-10:
                n_samples += 1
    
    A = np.zeros((n_samples, n))
    b = np.zeros(n_samples)
    idx = 0
    for (w0, wa) in models:
        for z in z_grid:
            if z <= 1e-10:
                continue
            z_nodes = z * nodes
            E_vals = _E_of_z_vec(z_nodes, w0, wa)
            f_vals = 1.0 / E_vals
            I_val = _true_luminosity_distance(z, w0, wa) / (1.0 + z)  # = integral
            A[idx, :] = f_vals
            b[idx] = I_val / z
            idx += 1
    
    # Solve least squares with non-negativity and sum=1 constraint
    # Use scipy.optimize for constrained problem
    from scipy.optimize import minimize as _min
    
    def _obj(w):
        # Max relative error
        err = 0.0
        idx2 = 0
        for (w0, wa) in models:
            for z in z_grid:
                if z <= 1e-10:
                    continue
                pred = z * np.dot(w, A[idx2, :])
                target = b[idx2]
                if target > 1e-15:
                    rel = abs(pred - target) / target
                    if rel > err:
                        err = rel
                idx2 += 1
        return err
    
    # Constraints: sum(w) = 1, w >= 0
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-10, None)] * n
    
    # Initial guess: equal weights
    w0_init = np.ones(n) / n
    
    result = _min(_obj, w0_init, method='SLSQP', bounds=bounds, constraints=cons,
                  options={'maxiter': 200, 'ftol': 1e-12})
    
    w = result.x
    # Ensure positivity and normalization
    w = np.maximum(w, 1e-15)
    w = w / w.sum()
    
    return w


def _objective(params, nodes_init, models, z_grid):
    """Objective for optimizer: worst relative error."""
    n = len(nodes_init)
    nodes = params[:n]
    weights = params[n:]
    
    # Enforce strictly increasing nodes in (0,1)
    if nodes[0] <= 0 or nodes[-1] >= 1:
        return 1e6
    for i in range(n - 1):
        if nodes[i+1] <= nodes[i]:
            return 1e6
    
    # Enforce positive weights
    if np.any(weights <= 0):
        return 1e6
    
    # Normalize weights
    w = weights / weights.sum()
    
    return _worst_rel_error(nodes, w, models, z_grid)


def _generate_node_families(n=8):
    """Generate diverse initial node configurations."""
    families = []
    
    # 1. Gauss-Legendre
    gl_nodes, _ = np.polynomial.legendre.leggauss(n)
    gl_nodes = (gl_nodes + 1) / 2  # map [-1,1] to [0,1]
    families.append(gl_nodes)
    
    # 2. Chebyshev nodes
    k = np.arange(n)
    cheb_nodes = 0.5 * (1 - np.cos(np.pi * (2*k + 1) / (2*n)))
    families.append(cheb_nodes)
    
    # 3. Log-spaced
    log_nodes = np.geomspace(1e-3, 1 - 1e-3, n)
    log_nodes = np.sort(log_nodes)
    families.append(log_nodes)
    
    # 4. Power-law (z^0.5)
    pow_nodes = np.linspace(0, 1, n+2)[1:-1] ** 0.5
    pow_nodes = np.sort(pow_nodes)
    families.append(pow_nodes)
    
    # 5. Power-law (z^2)
    pow_nodes2 = np.linspace(0, 1, n+2)[1:-1] ** 2
    pow_nodes2 = np.sort(pow_nodes2)
    families.append(pow_nodes2)
    
    # 6. Midpoint (current baseline)
    mid_nodes = (np.arange(n) + 0.5) / n
    families.append(mid_nodes)
    
    # 7. Clenshaw-Curtis
    cc_nodes = 0.5 * (1 - np.cos(np.pi * np.arange(n) / (n-1)))
    cc_nodes = np.sort(cc_nodes)
    families.append(cc_nodes)
    
    # 8. Random perturbations of GL
    rng = np.random.RandomState(42)
    for trial in range(5):
        pert = rng.uniform(-0.05, 0.05, n)
        rn = np.sort(gl_nodes + pert)
        rn = np.clip(rn, 1e-4, 1 - 1e-4)
        families.append(rn)
    
    return families


def construct():
    n = 8
    
    # Representative CPL models: (w0, wa)
    # Cover physically plausible range
    models = [
        (-1.0, 0.0),    # Cosmological constant
        (-1.0, 0.5),
        (-1.0, 1.0),
        (-0.9, 0.0),
        (-0.9, 0.5),
        (-0.9, 1.0),
        (-0.8, 0.0),
        (-0.8, 0.5),
        (-0.7, 0.0),
        (-0.7, 0.5),
        (-0.6, 0.0),
        (-0.5, 0.0),
        (-0.5, 0.5),
        (-0.3, 0.0),
        (-0.3, -0.5),
        (0.0, 0.0),     # Matter only
        (0.0, -0.5),
        (0.5, 0.0),
        (1.0, 0.0),
        (-1.2, 0.5),
        (-1.5, 0.0),
    ]
    
    # z grid for evaluation
    z_grid = np.array([
        0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0
    ])
    
    # Precompute true distances for speed
    true_dists = {}
    for (w0, wa) in models:
        for z in z_grid:
            true_dists[(w0, wa, z)] = _true_luminosity_distance(z, w0, wa)
    
    # Override to use precomputed values
    global _true_luminosity_distance
    _orig_true = _true_luminosity_distance
    
    def _true_luminosity_distance_fast(z, w0, wa):
        key = (w0, wa, z)
        if key in true_dists:
            return true_dists[key]
        return _orig_true(z, w0, wa)
    
    _true_luminosity_distance = _true_luminosity_distance_fast
    
    best_nodes = (np.arange(n) + 0.5) / n
    best_weights = np.full(n, 1.0 / n)
    best_err = _worst_rel_error(best_nodes, best_weights, models, z_grid)
    
    # Generate node families
    families = _generate_node_families(n)
    
    # Stage 1: Evaluate each family with optimized weights
    for fam_idx, nodes in enumerate(families):
        if _time_remaining() < 5.0:
            break
        
        # Ensure valid nodes
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        nodes = np.sort(nodes)
        # Ensure strictly increasing
        for i in range(n - 1):
            if nodes[i+1] - nodes[i] < 1e-8:
                nodes[i+1] = nodes[i] + 1e-6
        
        # Solve for weights
        weights = _solve_weights(nodes, models, z_grid)
        
        err = _worst_rel_error(nodes, weights, models, z_grid)
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    # Stage 2: Local refinement from best configuration
    if _time_remaining() > 10.0:
        # Parameterize: nodes (free) and log-weights (for positivity)
        def _obj_refine(params):
            nodes_r = params[:n]
            log_w = params[n:]
            weights_r = np.exp(log_w)
            weights_r = weights_r / weights_r.sum()
            
            if nodes_r[0] <= 0 or nodes_r[-1] >= 1:
                return 1e6
            for i in range(n - 1):
                if nodes_r[i+1] - nodes_r[i] < 1e-8:
                    return 1e6
            
            return _worst_rel_error(nodes_r, weights_r, models, z_grid)
        
        # Initial params
        init_params = np.concatenate([best_nodes, np.log(best_weights)])
        
        # Use Nelder-Mead for unconstrained optimization
        # with penalty for constraint violations
        def _obj_penalized(params):
            nodes_r = params[:n]
            log_w = params[n:]
            
            # Penalty for node constraints
            penalty = 0.0
            if nodes_r[0] <= 0:
                penalty += (nodes_r[0] - 0) ** 2 * 1e4
            if nodes_r[-1] >= 1:
                penalty += (nodes_r[-1] - 1) ** 2 * 1e4
            for i in range(n - 1):
                if nodes_r[i+1] - nodes_r[i] < 1e-8:
                    penalty += (1e-8 - (nodes_r[i+1] - nodes_r[i])) ** 2 * 1e4
            
            weights_r = np.exp(log_w)
            weights_r = weights_r / weights_r.sum()
            
            base_err = _worst_rel_error(nodes_r, weights_r, models, z_grid)
            return base_err + penalty
        
        result = minimize(_obj_penalized, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        if result.success or result.fun < best_err:
            nodes_r = np.clip(result.x[:n], 1e-6, 1 - 1e-6)
            nodes_r = np.sort(nodes_r)
            for i in range(n - 1):
                if nodes_r[i+1] - nodes_r[i] < 1e-8:
                    nodes_r[i+1] = nodes_r[i] + 1e-6
            
            log_w = result.x[n:]
            weights_r = np.exp(log_w)
            weights_r = np.maximum(weights_r, 1e-15)
            weights_r = weights_r / weights_r.sum()
            
            err = _worst_rel_error(nodes_r, weights_r, models, z_grid)
            if err < best_err:
                best_err = err
                best_nodes = nodes_r
                best_weights = weights_r
    
    # Final validation: ensure constraints are met
    best_nodes = np.sort(np.clip(best_nodes, 1e-6, 1 - 1e-6))
    for i in range(n - 1):
        if best_nodes[i+1] - best_nodes[i] < 1e-8:
            best_nodes[i+1] = best_nodes[i] + 1e-6
    
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / best_weights.sum()
    
    # Verify sum to 1 within 1e-12
    s = best_weights.sum()
    best_weights = best_weights / s
    
    # Restore original function
    _true_luminosity_distance = _orig_true
    
    return {"nodes": best_nodes, "weights": best_weights}


# Module-level: run construct at import time to cache results
# Actually, the evaluator calls construct(), so we just define it.
# But we need to make sure the function is callable and returns the right format.
