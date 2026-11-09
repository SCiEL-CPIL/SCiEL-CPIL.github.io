"""Model-and-optimize approach: Directly optimize nodes and weights using smooth least-squares for weights and Nelder-Mead for nodes."""
import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 88.0

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

def _solve_weights_ls(F, T):
    """Solve for weights minimizing sum of squared relative errors, with positivity constraint."""
    n_vars = 8
    n_tests = len(T)
    
    # Weighted least squares: minimize sum_j (F[j,:] @ w - T[j])^2 / T[j]^2
    # subject to sum(w) = 1, w >= 0
    # Use constrained least squares via scipy
    W = 1.0 / T  # weights for each test point (relative error weighting)
    W[W < 1e10] = 0  # skip zero or near-zero T
    
    # Build weighted LS problem
    Fw = F * W[:, None]
    Tw = T * W
    
    # Minimize ||Fw @ w - Tw||^2 s.t. sum(w)=1, w>=0
    # Use scipy.optimize.minimize with SLSQP for small problem
    def obj(w):
        r = Fw @ w - Tw
        return np.dot(r, r)
    
    def grad(w):
        r = Fw @ w - Tw
        return 2.0 * Fw.T @ r
    
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
    bounds = [(0, None)] * 8
    
    # Initial guess: uniform
    w0 = np.ones(8) / 8.0
    
    try:
        res = minimize(obj, w0, method='SLSQP', jac=grad, bounds=bounds, constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-14})
        if res.success and res.x is not None:
            w = res.x
            w = np.maximum(w, 0)
            w = w / np.sum(w)
            # Compute worst-case relative error
            errors = np.abs(F @ w - T) / T
            errors = errors[np.isfinite(errors)]
            return np.max(errors) if len(errors) > 0 else 1e10
        else:
            return 1e10
    except Exception:
        return 1e10

def _solve_weights_lp(F, T):
    """Try LP first, fall back to LS if infeasible."""
    n_vars = 8
    n_tests = len(T)
    c = np.zeros(n_vars + 1)
    c[-1] = 1.0
    A_ub = np.zeros((2 * n_tests, n_vars + 1))
    b_ub = np.zeros(2 * n_tests)
    for j in range(n_tests):
        t_j = T[j]
        if t_j < 1e-10:
            continue
        A_ub[2*j, :n_vars] = F[j, :]
        A_ub[2*j, -1] = -t_j
        b_ub[2*j] = t_j
        A_ub[2*j+1, :n_vars] = -F[j, :]
        A_ub[2*j+1, -1] = t_j
        b_ub[2*j+1] = t_j
    A_eq = np.zeros((1, n_vars + 1))
    A_eq[0, :n_vars] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n_vars + [(0, None)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[n_vars]
        else:
            return _solve_weights_ls(F, T)
    except Exception:
        return _solve_weights_ls(F, T)

def _objective_nodes(params):
    if _time_left() < 2.0:
        return 1e10
    nodes_raw = 1.0 / (1.0 + np.exp(-params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6

    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8)
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    F, T = _compute_test_data(nodes, test_cosmos, test_zs)
    return _solve_weights_lp(F, T)

def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    idx = np.argsort(nodes)
    return nodes[idx]

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

def _joint_objective(params):
    """Joint optimization of nodes and weights using smooth surrogate."""
    if _time_left() < 2.0:
        return 1e10
    # First 8 params: node locations (sigmoid encoded)
    # Last 8 params: weights (softmax encoded for positivity, normalized)
    node_params = params[:8]
    weight_params = params[8:]
    
    nodes = _decode_params(node_params)
    
    # Softmax weights
    w_exp = np.exp(weight_params - np.max(weight_params))
    weights = w_exp / np.sum(w_exp)
    
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8)
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    F, T = _compute_test_data(nodes, test_cosmos, test_zs)
    
    # Compute relative errors
    approx = F @ weights
    rel_errors = np.abs(approx - T) / T
    rel_errors = rel_errors[np.isfinite(rel_errors)]
    
    if len(rel_errors) == 0:
        return 1e10
    
    # Soft-max surrogate for worst-case error
    alpha = 50.0
    max_err = np.max(rel_errors)
    if max_err < 1e-15:
        return 0.0
    log_sum_exp = np.log(np.sum(np.exp(alpha * rel_errors))) / alpha
    return log_sum_exp

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 88.0
    
    gl_nodes = _gauss_legendre_init()
    params_init = _encode_initial(gl_nodes)
    best_params = params_init.copy()
    best_score = _objective_nodes(best_params)
    
    np.random.seed(42)
    restarts = 5
    for i in range(restarts):
        if _time_left() < 10.0:
            break
        if i == 0:
            init_p = params_init.copy()
        else:
            init_p = params_init + np.random.randn(8) * 0.5
        try:
            res = minimize(_objective_nodes, init_p, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    nodes_raw = 1.0 / (1.0 + np.exp(-best_params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6

    # Joint polish: optimize nodes and weights together using smooth surrogate
    if _time_left() > 15.0:
        # Initialize weights from LS solution
        test_cosmos_init = [
            (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
            (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
            (-0.2, 0.2), (0.8, -0.8)
        ]
        test_zs_init = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        F_init, T_init = _compute_test_data(nodes, test_cosmos_init, test_zs_init)
        
        # Get initial weights from LS
        w_ls = np.ones(8) / 8.0
        try:
            W = 1.0 / T_init
            W[W < 1e10] = 0
            Fw = F_init * W[:, None]
            Tw = T_init * W
            def obj_w(w):
                r = Fw @ w - Tw
                return np.dot(r, r)
            def grad_w(w):
                r = Fw @ w - Tw
                return 2.0 * Fw.T @ r
            cons_w = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
            bounds_w = [(0, None)] * 8
            res_w = minimize(obj_w, w_ls, method='SLSQP', jac=grad_w, bounds=bounds_w, constraints=cons_w,
                           options={'maxiter': 500, 'ftol': 1e-14})
            if res_w.success:
                w_ls = np.maximum(res_w.x, 0)
                w_ls = w_ls / np.sum(w_ls)
        except Exception:
            pass
        
        # Encode weights for joint optimization
        w_enc = np.log(w_ls + 1e-15)
        joint_init = np.concatenate([best_params, w_enc])
        
        try:
            res_joint = minimize(_joint_objective, joint_init, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12})
            if res_joint.fun < best_score:
                # Extract improved nodes
                best_params = res_joint.x[:8]
                nodes_raw = 1.0 / (1.0 + np.exp(-best_params))
                nodes = np.sort(nodes_raw)
                nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
                for i in range(1, 8):
                    if nodes[i] <= nodes[i-1]:
                        nodes[i] = nodes[i-1] + 1e-6
        except Exception:
            pass

    final_test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1),
        (0.2, -0.5), (-0.8, 0.2)
    ]
    final_test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    F_final, T_final = _compute_test_data(nodes, final_test_cosmos, final_test_zs)
    
    # Try LP for final weights, fall back to LS
    n_vars = 8
    n_tests = len(T_final)
    c = np.zeros(n_vars + 1)
    c[-1] = 1.0
    A_ub = np.zeros((2 * n_tests, n_vars + 1))
    b_ub = np.zeros(2 * n_tests)
    for j in range(n_tests):
        t_j = T_final[j]
        if t_j < 1e-10:
            continue
        A_ub[2*j, :n_vars] = F_final[j, :]
        A_ub[2*j, -1] = -t_j
        b_ub[2*j] = t_j
        A_ub[2*j+1, :n_vars] = -F_final[j, :]
        A_ub[2*j+1, -1] = t_j
        b_ub[2*j+1] = t_j
    A_eq = np.zeros((1, n_vars + 1))
    A_eq[0, :n_vars] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n_vars + [(0, None)]
    
    weights = None
    try:
        res_final = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res_final.status == 0:
            weights = res_final.x[:n_vars]
    except Exception:
        pass
    
    if weights is None:
        # Fall back to LS
        try:
            W = 1.0 / T_final
            W[W < 1e10] = 0
            Fw = F_final * W[:, None]
            Tw = T_final * W
            def obj_final(w):
                r = Fw @ w - Tw
                return np.dot(r, r)
            def grad_final(w):
                r = Fw @ w - Tw
                return 2.0 * Fw.T @ r
            cons_final = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
            bounds_final = [(0, None)] * 8
            w0_final = np.ones(8) / 8.0
            res_ls = minimize(obj_final, w0_final, method='SLSQP', jac=grad_final, bounds=bounds_final, constraints=cons_final,
                             options={'maxiter': 1000, 'ftol': 1e-16})
            if res_ls.success:
                weights = np.maximum(res_ls.x, 0)
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(8) / 8.0
        except Exception:
            weights = np.ones(8) / 8.0
    
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
