"""
Strategy: Continuous Game-Theoretic Robust Optimization via Frank-Wolfe / Mirror Descent.

Key Structural Change:
Instead of discretizing the cosmology space into a fixed grid of test points and solving
a Linear Program (LP) for weights at each node configuration (which is slow and biased
to the grid), we treat the problem as a zero-sum game:
  Minimize over (nodes, weights): Maximize over (z, w0, wa): Relative Error.

We use a Frank-Wolfe / Mirror Descent approach:
1. We maintain a "distribution" over cosmologies (the dual variables / adversary's mixed strategy).
2. For fixed nodes, we update weights using a projected subgradient step towards the
   worst-case direction defined by the current adversary distribution.
3. We update nodes using a subgradient step on the minimax objective.
4. We update the adversary's distribution by adding the current worst-case point (found
   via a local search in the continuous parameter space) to the support.

This avoids the LP overhead entirely and directly targets the continuous worst-case error.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, roots_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0 for CPL dark energy model."""
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)

def _reference_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference: \int_0^1 dt / E(z*t)."""
    if z < 1e-10:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, err = quad(integrand, 0, 1, limit=300, epsabs=1e-13, epsrel=1e-13)
    return val

def _quadrature_value(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute \sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)

def _find_worst_case(nodes, weights, Omega_m=0.3):
    """
    Find the (z, w0, wa) that maximizes the relative error for given nodes and weights.
    Uses a local optimization from multiple starting points to approximate the global max.
    """
    def _neg_rel_err(params):
        z, w0, wa = params
        if z < 1e-6:
            return 0.0
        ref = _reference_integral(z, w0, wa, Omega_m)
        q_val = _quadrature_value(nodes, weights, z, w0, wa, Omega_m)
        rel_err = abs(q_val - ref) / ref
        return -rel_err
    
    # Starting points covering the parameter space
    starts = [
        [0.5, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.1, 0.5, -0.5],
        [5.0, -1.5, 1.0],
        [0.3, -0.5, 0.5],
        [1.5, 0.0, -1.0],
        [0.8, 1.0, 1.0],
    ]
    
    best_err = -1.0
    best_params = starts[0]
    
    for start in starts:
        try:
            res = minimize(_neg_rel_err, start, method='Nelder-Mead',
                           bounds=([(1e-6, 10), (-2, 2), (-2, 2)]),
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Return the worst case parameters and the actual error (positive)
    z, w0, wa = best_params
    ref = _reference_integral(z, w0, wa, Omega_m)
    q_val = _quadrature_value(nodes, weights, z, w0, wa, Omega_m)
    rel_err = abs(q_val - ref) / ref
    return (z, w0, wa), rel_err

def _gauss_legendre_nodes(n=8):
    nodes, _ = roots_legendre(n)
    return (nodes + 1.0) / 2.0

def construct():
    if time.time() > _DEADLINE - 5:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # Initialize nodes and weights
    nodes = _gauss_legendre_nodes(8)
    weights = np.ones(8) / 8.0
    
    # Dual variables: a set of "active" cosmologies and their weights in the adversary's mix
    # We start with a diverse set
    active_cosmos = [
        (0.5, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.1, 0.5, -0.5),
        (5.0, -1.5, 1.0),
        (0.3, -0.5, 0.5),
    ]
    dual_weights = np.ones(len(active_cosmos)) / len(active_cosmos)
    
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = float('inf')
    
    # Frank-Wolfe / Mirror Descent iterations
    # We alternate between updating (nodes, weights) and updating the adversary's distribution
    
    # Pre-compute reference integrals for active cosmologies
    refs = np.array([_reference_integral(z, w0, wa) for z, w0, wa in active_cosmos])
    
    # Step size decay
    step_nodes = 0.05
    step_weights = 0.1
    
    for it in range(300):
        if time.time() > _DEADLINE - 10:
            break
            
        # 1. Evaluate current worst-case error and find new worst-case point
        worst_params, current_err = _find_worst_case(nodes, weights)
        
        if current_err < best_err:
            best_err = current_err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
            
        # Check for convergence
        if current_err < 1e-6:
            break
            
        # 2. Update Adversary's Distribution (Dual Ascent)
        # Add the new worst-case point to the active set if it's not already there
        # (Simplified: just add it and renormalize, or replace the least important one)
        # For efficiency, we keep a fixed size active set by replacing the one with lowest dual weight
        if len(active_cosmos) >= 20:
            # Find index with smallest dual weight
            min_idx = np.argmin(dual_weights)
            active_cosmos[min_idx] = worst_params
            refs[min_idx] = _reference_integral(*worst_params)
            dual_weights[min_idx] = 0.1 # Reset weight
        else:
            active_cosmos.append(worst_params)
            refs = np.append(refs, _reference_integral(*worst_params))
            dual_weights = np.append(dual_weights, 0.1)
            
        # Normalize dual weights
        dual_weights /= np.sum(dual_weights)
        
        # 3. Compute Subgradients for Nodes and Weights
        # The objective is max_i (dual_weights[i] * rel_err_i)
        # rel_err_i = |q_i - ref_i| / ref_i
        # q_i = sum_j w_j / E(z_i * t_j)
        
        # We need to compute the gradient of the "expected" relative error w.r.t nodes and weights.
        # This is complex. A simpler approach for the "primal" update in Frank-Wolfe is to
        # move towards a vertex of the feasible region that minimizes the linear approximation.
        
        # However, implementing full Frank-Wolfe for nodes (simplex-like but with ordering constraints)
        # is tricky. Let's use a subgradient method on the nodes and a projected gradient on weights.
        
        # Compute gradients
        grad_nodes = np.zeros(8)
        grad_weights = np.zeros(8)
        
        # We approximate the subgradient by sampling from the active set
        # Actually, for speed, let's just use the single worst-case point for the gradient step
        # This is a heuristic "steepest descent" on the worst-case surface.
        
        z, w0, wa = worst_params
        ref = _reference_integral(z, w0, wa)
        
        # q = sum w_j / E(z*t_j)
        # dq/dw_j = 1/E(z*t_j)
        # dq/dt_j = w_j * d(1/E)/dt_j = w_j * (-1/E^2) * dE/dt_j
        # dE/dt_j = z * dE/dz
        
        E_vals = _E_of_z(z * nodes, w0, wa)
        inv_E = 1.0 / E_vals
        
        # dE/dz calculation
        # E = sqrt(Om (1+z)^3 + Ode (1+z)^(3(1+w0+wa z)))
        # dE/dz = 1/(2E) * [ 3 Om (1+z)^2 + Ode * 3(1+w0+wa z) (1+z)^(3(1+w0+wa z)-1) + Ode * 3 wa (1+z)^(3(1+w0+wa z)) ]
        # This is getting messy. Let's use numerical differentiation for the node gradient.
        
        h = 1e-5
        for j in range(8):
            nodes_plus = nodes.copy()
            nodes_plus[j] += h
            # Ensure ordering
            if j < 7 and nodes_plus[j] >= nodes_plus[j+1]:
                nodes_plus[j] = nodes_plus[j+1] - 1e-6
            if j > 0 and nodes_plus[j] <= nodes_plus[j-1]:
                nodes_plus[j] = nodes_plus[j-1] + 1e-6
            
            q_plus = _quadrature_value(nodes_plus, weights, z, w0, wa)
            dq_dtj = (q_plus - _quadrature_value(nodes, weights, z, w0, wa)) / h
            # d(q-ref)/dt_j = dq_dtj * sign(q-ref)
            grad_nodes[j] = dq_dtj * np.sign(_quadrature_value(nodes, weights, z, w0, wa) - ref)
            
        # Gradient w.r.t weights is simpler
        q_val = _quadrature_value(nodes, weights, z, w0, wa)
        grad_weights = inv_E * np.sign(q_val - ref)
        
        # Normalize gradients
        norm_gn = np.linalg.norm(grad_nodes)
        if norm_gn > 1e-12:
            grad_nodes /= norm_gn
        norm_gw = np.linalg.norm(grad_weights)
        if norm_gw > 1e-12:
            grad_weights /= norm_gw
            
        # 4. Update Nodes (Subgradient Descent)
        # Move opposite to gradient
        nodes_new = nodes - step_nodes * grad_nodes
        # Project to [0,1] and enforce ordering
        nodes_new = np.clip(nodes_new, 1e-4, 1 - 1e-4)
        nodes_new = np.sort(nodes_new)
        # Ensure strict increase
        for j in range(1, 8):
            if nodes_new[j] <= nodes_new[j-1]:
                nodes_new[j] = nodes_new[j-1] + 1e-6
        nodes = nodes_new
        
        # 5. Update Weights (Projected Gradient Descent)
        weights_new = weights - step_weights * grad_weights
        # Project to simplex: sum=1, w>=0
        # Simple projection: clip to 0 and renormalize
        weights_new = np.clip(weights_new, 0, None)
        sum_w = np.sum(weights_new)
        if sum_w < 1e-12:
            weights_new = np.ones(8) / 8.0
        else:
            weights_new /= sum_w
        weights = weights_new
        
        # Decay step sizes
        step_nodes *= 0.98
        step_weights *= 0.98
        
    # Final verification and cleanup
    w_sum = np.sum(weights)
    if w_sum > 1e-12:
        weights = weights / w_sum
        
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    
    diffs = np.diff(best_nodes)
    if np.any(diffs <= 0):
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    if np.any(best_weights <= 0):
        # If weights are invalid, fall back to current optimized weights if valid, else GL
        if np.all(weights > 0):
            best_weights = weights
        else:
            nodes, weights = roots_legendre(8)
            nodes = (nodes + 1) / 2
            weights = weights / 2
            return {"nodes": nodes, "weights": weights}
    
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
