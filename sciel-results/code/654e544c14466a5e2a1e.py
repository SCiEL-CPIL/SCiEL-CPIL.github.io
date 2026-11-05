"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Direct global minimax optimization.
We treat the problem as finding a quadrature rule that minimizes the worst-case
relative error across a diverse set of cosmological models.
1. Build a library of challenging models (varying w0, wa, Om).
2. For a given set of nodes, the optimal weights are found via Linear Programming
   (minimize t s.t. |quad - ref| <= t * ref).
3. Optimize the nodes using a multi-start Nelder-Mead search, where the objective
   is the worst-case error obtained from the LP-optimal weights for those nodes.
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog

START = time.time()
DEADLINE = 85.0  # 5s margin

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa, Om):
    """
    E(a) for flat CPL universe.
    H^2/H0^2 = Om a^-3 + Ol(a)
    Ol(a) = Ol0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
    Ol0 = 1 - Om
    """
    a = np.clip(a, 1e-8, 1.0)
    Ol0 = 1.0 - Om
    # Avoid issues if Ol0 <= 0 (shouldn't happen for valid cosmologies)
    if Ol0 <= 0:
        return np.sqrt(a**(-3.0))
    
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * (1.0 + w0 + wa))
    Ol = Ol0 * pow_term * exp_term
    
    E2 = a ** (-3.0) + Ol
    # Ensure positive
    E2 = np.maximum(E2, 1e-10)
    return np.sqrt(E2)

def _build_library():
    """
    Create a diverse set of cosmologies to test against.
    We want to cover the range of behaviors that might appear in validation.
    """
    models = []
    # Vary w0, wa, Om
    w0_vals = [-1.5, -1.0, -0.5, 0.0]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    Om_vals = [0.2, 0.3, 0.4]
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for Om in Om_vals:
                models.append((w0, wa, Om))
                
    # Add some specific edge cases or common models
    # LambdaCDM
    models.append((-1.0, 0.0, 0.3))
    # Phantom
    models.append((-1.2, 0.5, 0.3))
    # Quintessence
    models.append((-0.5, 0.5, 0.3))
    
    return models

def _get_error(nodes, weights, zs, models):
    """
    Compute worst-case relative error for the given nodes/weights.
    """
    max_err = 0.0
    for (w0, wa, Om) in models:
        Efun = lambda a: _E_cpl(a, w0, wa, Om)
        for z in zs:
            # Quadrature: (1+z) * z * sum(w_i / E(a_i))
            a = 1.0 / (1.0 + z * nodes)
            E = Efun(a)
            quad_val = (1.0 + z) * z * np.sum(weights / E)
            
            # Reference: High-order Gauss-Legendre
            # Integral I = \int_0^z dz' / E(1/(1+z'))
            # Change var u in [0,1], z' = z u, dz' = z du
            # I = z * \int_0^1 du / E(1/(1+z u))
            n_ref = 64
            xi, wi = np.polynomial.legendre.leggauss(n_ref)
            u = 0.5 * (xi + 1.0)
            a_ref = 1.0 / (1.0 + z * u)
            E_ref = Efun(a_ref)
            ref_val = (1.0 + z) * z * np.sum(wi / E_ref)
            
            if ref_val == 0:
                continue
            err = abs(quad_val - ref_val) / ref_val
            if err > max_err:
                max_err = err
    return max_err

def _optimize_weights(nodes, zs, models):
    """
    Optimize weights for fixed nodes using LP.
    Minimize t such that |quad_i - ref_i| <= t * ref_i for all test cases i.
    """
    n_nodes = len(nodes)
    c = np.zeros(n_nodes + 1)
    c[-1] = 1.0  # Minimize t
    
    A_ub_list = []
    b_ub_list = []
    
    for (w0, wa, Om) in models:
        Efun = lambda a: _E_cpl(a, w0, wa, Om)
        for z in zs:
            a = 1.0 / (1.0 + z * nodes)
            E = Efun(a)
            A_row = (1.0 + z) * z / E
            
            # Reference
            n_ref = 32 # Use fewer points for LP speed, still accurate enough for constraint
            xi, wi = np.polynomial.legendre.leggauss(n_ref)
            u = 0.5 * (xi + 1.0)
            a_ref = 1.0 / (1.0 + z * u)
            E_ref = Efun(a_ref)
            ref_val = (1.0 + z) * z * np.sum(wi / E_ref)
            
            if ref_val == 0:
                continue
            
            # Constraint 1: sum(A_j w_j) - t * ref <= ref
            # sum(A_j w_j) - t * ref - ref <= 0
            row1 = np.zeros(n_nodes + 1)
            row1[:n_nodes] = A_row
            row1[-1] = -ref_val
            A_ub_list.append(row1)
            b_ub_list.append(ref_val)
            
            # Constraint 2: -sum(A_j w_j) + t * ref <= -ref
            # -sum(A_j w_j) + t * ref + ref <= 0
            row2 = np.zeros(n_nodes + 1)
            row2[:n_nodes] = -A_row
            row2[-1] = ref_val
            A_ub_list.append(row2)
            b_ub_list.append(-ref_val)
            
    if not A_ub_list:
        return np.ones(n_nodes) / n_nodes

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    bounds = [(0, None)] * n_nodes + [(0, None)]
    
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status == 0:
        weights = res.x[:n_nodes]
        weights = np.clip(weights, 1e-12, None)
        weights = weights / np.sum(weights)
        return weights
    else:
        return np.ones(n_nodes) / n_nodes

def _node_objective(x, zs, models):
    """
    Objective function for node optimization.
    x: nodes
    """
    if x[0] < 1e-5 or x[-1] > 1 - 1e-5:
        return 1e10
    if np.any(np.diff(x) < 1e-5):
        return 1e10
    
    # Get optimal weights for these nodes
    weights = _optimize_weights(x, zs, models)
    # Evaluate error with these weights
    err = _get_error(x, weights, zs, models)
    return err

def construct():
    fallback = _fallback()
    
    # Build library
    models = _build_library()
    
    # Redshifts to test
    # Focus on z=0.1 to 3.0, which is where structure is most complex
    zs = np.linspace(0.1, 3.0, 20)
    
    # Initial nodes: Gauss-Legendre
    xi, _ = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (xi + 1.0)
    nodes_init = np.sort(nodes_init)
    
    best_nodes = nodes_init.copy()
    best_weights = _optimize_weights(best_nodes, zs, models)
    best_err = _get_error(best_nodes, best_weights, zs, models)
    
    # Multi-start Nelder-Mead
    # We will try a few different starting points
    starts = [
        nodes_init,
        np.linspace(0.05, 0.95, 8),
        np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]), # Just to have variety, though GL is usually better
    ]
    
    # Add random perturbations to GL nodes
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = nodes_init + rng.normal(0, 0.05, 8)
        perturbed = np.sort(perturbed)
        # Clip to [0,1] and ensure strict increase
        perturbed[0] = max(perturbed[0], 1e-4)
        perturbed[-1] = min(perturbed[-1], 1-1e-4)
        # Ensure spacing
        for i in range(1, 8):
            if perturbed[i] <= perturbed[i-1] + 1e-4:
                perturbed[i] = perturbed[i-1] + 1e-4
        if perturbed[-1] <= 1-1e-4:
            starts.append(perturbed)
    
    for start in starts:
        if time.time() > DEADLINE - 10.0:
            break
            
        def objective(x):
            return _node_objective(x, zs, models)
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.success:
                nodes = res.x
                # Validate
                if nodes[0] > 1e-5 and nodes[-1] < 1 - 1e-5 and np.all(np.diff(nodes) > 1e-5):
                    weights = _optimize_weights(nodes, zs, models)
                    err = _get_error(nodes, weights, zs, models)
                    if err < best_err:
                        best_err = err
                        best_nodes = nodes.copy()
                        best_weights = weights.copy()
        except Exception:
            pass
            
    # Final check and normalization
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
