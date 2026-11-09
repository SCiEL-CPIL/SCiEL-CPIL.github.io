"""Chebyshev-based Remez-like optimization for universal quadrature rule."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

_START_TIME = time.time()
_DEADLINE = _START_TIME + 86.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

# Precompute high-accuracy Gauss-Legendre nodes/weights for reference integration
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    if z == 0:
        return 0.0
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

def _compute_test_data(nodes, test_cosmos, test_zs):
    num_tests = len(test_cosmos) * len(test_zs)
    F = np.zeros((num_tests, 8))
    T = np.zeros(num_tests)
    
    idx = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z == 0:
                T[idx] = 1.0
                F[idx, :] = 1.0
            else:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa)
                F[idx, :] = 1.0 / E_vals
                true_val = _true_dL(z, w0, wa)
                T[idx] = true_val / ((1.0 + z) * z)
            idx += 1
    return F, T

def _solve_weights_lp_minimax(F, T, active_indices=None):
    """
    Solve for weights minimizing max relative error on active set.
    Variables: w_0..w_7, epsilon
    Minimize epsilon
    Subject to:
      -epsilon <= (F_i @ w - T_i) / T_i <= epsilon
      sum(w) = 1
      w >= 0
    """
    if active_indices is None:
        active_indices = np.arange(len(T))
        
    F_act = F[active_indices]
    T_act = T[active_indices]
    
    # Relative error formulation:
    # (F_i @ w - T_i) / T_i <= epsilon  =>  (F_i / T_i) @ w - 1 <= epsilon
    # -(F_i @ w - T_i) / T_i <= epsilon =>  -(F_i / T_i) @ w + 1 <= epsilon
    # Let A_rel = F_act / T_act[:, None]
    # Constraints:
    # A_rel @ w - epsilon <= 1
    # -A_rel @ w - epsilon <= 1
    
    A_rel = F_act / T_act[:, None]
    n_constraints = 2 * len(active_indices)
    n_vars = 9 # 8 weights + 1 epsilon
    
    # c = [0, 0, ..., 0, 1] (minimize epsilon)
    c = np.zeros(n_vars)
    c[-1] = 1.0
    
    # A_ub, b_ub
    # Rows 1..N: A_rel @ w - epsilon <= 1
    # Rows N+1..2N: -A_rel @ w - epsilon <= 1
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.ones(n_constraints)
    
    # First N rows
    A_ub[:len(active_indices), :8] = A_rel
    A_ub[:len(active_indices), 8] = -1.0
    
    # Next N rows
    A_ub[len(active_indices):, :8] = -A_rel
    A_ub[len(active_indices):, 8] = -1.0
    
    # A_eq, b_eq
    # sum(w) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :8] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds
    # w_i >= 0, epsilon >= 0 (epsilon can be negative in theory but we minimize it, so it will be >= 0 if feasible, or we can bound it)
    # Actually, if the rule is perfect, epsilon can be 0. If not, it's positive.
    # We can allow epsilon to be any real number, but we minimize it.
    # However, if we set epsilon >= 0, we force it to be non-negative.
    # Let's allow epsilon to be free, but the constraints will push it up.
    # But linprog requires bounds. Let's set epsilon >= -1 (safe lower bound) and w >= 0.
    bounds = [(0, None)] * 8 + [(-1.0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            w = res.x[:8]
            w = np.maximum(w, 0)
            w = w / np.sum(w)
            return w, res.x[-1]
        else:
            return None, np.inf
    except Exception:
        return None, np.inf

def _objective_nodes(params):
    if _time_left() < 2.0:
        return 1e10
    
    nodes_raw = 1.0 / (1.0 + np.exp(-params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6

    # Test set for active set selection
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1)
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    F, T = _compute_test_data(nodes, test_cosmos, test_zs)
    
    # Initial active set: all points
    active_indices = np.arange(len(T))
    
    # Iterative Remez-like loop
    best_w = None
    best_eps = np.inf
    
    for _ in range(5):
        if _time_left() < 2.0: break
        w, eps = _solve_weights_lp_minimax(F, T, active_indices)
        if w is None:
            break
        if eps < best_eps:
            best_eps = eps
            best_w = w
            
        # Update active set: find top K errors with current w
        approx = F @ w
        rel_errors = np.abs(approx - T) / T
        # Select top 20 errors
        top_k = 20
        if len(rel_errors) > top_k:
            active_indices = np.argsort(rel_errors)[-top_k:]
        else:
            active_indices = np.arange(len(rel_errors))
            
        # If active set didn't change or eps is very small, break
        # Simple check: if max error on full set is close to eps, we are good
        if len(rel_errors) > 0:
            full_max_err = np.max(rel_errors)
            if abs(full_max_err - eps) < 1e-6:
                break
                
    if best_w is None:
        return 1e10
        
    # Calculate final worst error on full test set
    approx = F @ best_w
    rel_errors = np.abs(approx - T) / T
    return np.max(rel_errors)

def _chebyshev_init():
    """Chebyshev roots mapped to [0, 1]."""
    k = np.arange(8)
    x_cheb = np.cos((2*k + 1) * np.pi / 16.0)
    nodes = 0.5 * (1.0 - x_cheb)
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    return nodes

def _encode_initial(nodes):
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    return -np.log(1.0/nodes - 1.0)

def _decode_params(params):
    nodes_raw = 1.0 / (1.0 + np.exp(-params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    return nodes

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 86.0
    
    # Initialize with Chebyshev nodes
    init_nodes = _chebyshev_init()
    init_params = _encode_initial(init_nodes)
    
    best_params = init_params.copy()
    best_score = _objective_nodes(init_params)
    
    # Nelder-Mead optimization on nodes
    # We use the Remez-like weight solver inside the objective
    np.random.seed(42)
    
    restarts = 3
    for i in range(restarts):
        if _time_left() < 15.0:
            break
        if i == 0:
            init_p = best_params.copy()
        else:
            # Perturb from best
            init_p = best_params + np.random.randn(8) * 0.5
            
        try:
            res = minimize(_objective_nodes, init_p, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final node decoding
    nodes = _decode_params(best_params)
    
    # Final weight calculation using the LP minimax on a larger, disjoint-like test set
    # Note: The task says "disjoint public validation cosmologies". 
    # We should ensure our training set is different from validation if possible, 
    # but we don't have access to validation. We use a broad representative set.
    
    final_test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1),
        (0.2, -0.5), (-0.8, 0.2), (0.0, 0.5), (-0.5, -0.5)
    ]
    final_test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    F_final, T_final = _compute_test_data(nodes, final_test_cosmos, final_test_zs)
    
    # Solve weights for final set
    weights, _ = _solve_weights_lp_minimax(F_final, T_final, active_indices=np.arange(len(T_final)))
    
    if weights is None:
        # Fallback to uniform
        weights = np.ones(8) / 8.0
    else:
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
    return {"nodes": nodes, "weights": weights}
