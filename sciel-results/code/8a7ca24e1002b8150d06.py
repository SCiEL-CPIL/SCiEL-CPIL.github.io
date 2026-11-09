"""Model-and-optimize approach: Directly optimize nodes and weights using smooth least-squares for weights and Nelder-Mead for nodes."""
import time
import numpy as np
from scipy.optimize import minimize

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

def _solve_weights_ls(F, T):
    """Solve for weights minimizing sum of squared relative errors, with positivity constraint."""
    n_vars = 8
    n_tests = len(T)
    
    W = 1.0 / T
    W[W < 1e10] = 0
    
    Fw = F * W[:, None]
    Tw = T * W
    
    def obj(w):
        r = Fw @ w - Tw
        return np.dot(r, r)
    
    def grad(w):
        r = Fw @ w - Tw
        return 2.0 * Fw.T @ r
    
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
    bounds = [(0, None)] * 8
    
    w0 = np.ones(8) / 8.0
    
    try:
        res = minimize(obj, w0, method='SLSQP', jac=grad, bounds=bounds, constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-14})
        if res.success and res.x is not None:
            w = res.x
            w = np.maximum(w, 0)
            w = w / np.sum(w)
            errors = np.abs(F @ w - T) / T
            errors = errors[np.isfinite(errors)]
            return np.max(errors) if len(errors) > 0 else 1e10
        else:
            return 1e10
    except Exception:
        return 1e10

def _solve_weights_minimax(F, T, alpha=20.0):
    """Solve for weights minimizing a smooth surrogate of max relative error."""
    n_vars = 8
    n_tests = len(T)
    
    W = 1.0 / T
    W[W < 1e10] = 0
    
    Fw = F * W[:, None]
    Tw = T * W
    
    def obj(w):
        r = Fw @ w - Tw
        # Soft max of |r|
        abs_r = np.abs(r)
        max_r = np.max(abs_r)
        if max_r < 1e-15:
            return 0.0
        scaled = alpha * abs_r / max_r
        log_sum_exp = np.log(np.sum(np.exp(scaled))) / alpha * max_r
        return log_sum_exp
    
    def grad(w):
        r = Fw @ w - Tw
        abs_r = np.abs(r)
        max_r = np.max(abs_r)
        if max_r < 1e-15:
            return np.zeros(n_vars)
        scaled = alpha * abs_r / max_r
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        # d/dw of log_sum_exp * max_r
        # log_sum_exp = log(sum(exp(scaled))) / alpha * max_r
        # d(log_sum_exp)/dw_j = (1/alpha) * (sum_i exp(scaled_i) * alpha * sign(r_i) * Fw[i,j] / max_r) / sum_exp * max_r
        # = (sum_i exp(scaled_i) * sign(r_i) * Fw[i,j]) / sum_exp
        signs = np.sign(r)
        weights_vec = exp_scaled / sum_exp
        grad_out = Fw.T @ (weights_vec * signs)
        return grad_out
    
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
    bounds = [(0, None)] * 8
    
    w0 = np.ones(8) / 8.0
    
    try:
        res = minimize(obj, w0, method='SLSQP', jac=grad, bounds=bounds, constraints=cons,
                       options={'maxiter': 800, 'ftol': 1e-15})
        if res.success and res.x is not None:
            w = res.x
            w = np.maximum(w, 0)
            w = w / np.sum(w)
            errors = np.abs(F @ w - T) / T
            errors = errors[np.isfinite(errors)]
            return np.max(errors) if len(errors) > 0 else 1e10
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
    return _solve_weights_minimax(F, T)

def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    idx = np.argsort(nodes)
    return nodes[idx]

def _chebyshev_init():
    """Chebyshev root-based initialization mapped to account for non-uniform integrand behavior.
    
    The integrand 1/E(a) varies more rapidly at high z (small a). We use a
    power-law remapping of Chebyshev nodes to cluster more nodes at high z.
    """
    # Chebyshev roots on [-1, 1]
    k = np.arange(8)
    x_cheb = np.cos((2*k + 1) * np.pi / 16.0)
    # Map to [0, 1]
    nodes = 0.5 * (1.0 - x_cheb)
    nodes = np.sort(nodes)
    # Apply a power-law warp to cluster nodes toward higher z (lower a)
    # The integrand 1/E(a) ~ a^3 for matter-dominated, so more weight at small a
    # Use a mild warp: t -> t^p where p < 1 clusters toward t=0 (high z)
    # But our nodes are in z-space [0,1], so we want more nodes at high z
    p = 0.85
    nodes = nodes ** p
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
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

def _joint_objective(params):
    """Joint optimization of nodes and weights using smooth surrogate."""
    if _time_left() < 2.0:
        return 1e10
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
    
    approx = F @ weights
    rel_errors = np.abs(approx - T) / T
    rel_errors = rel_errors[np.isfinite(rel_errors)]
    
    if len(rel_errors) == 0:
        return 1e10
    
    alpha = 50.0
    max_err = np.max(rel_errors)
    if max_err < 1e-15:
        return 0.0
    log_sum_exp = np.log(np.sum(np.exp(alpha * rel_errors))) / alpha
    return log_sum_exp

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 86.0
    
    # Multiple initializations: Gauss-Legendre and Chebyshev-warped
    gl_nodes = _gauss_legendre_init()
    cheb_nodes = _chebyshev_init()
    
    best_params = None
    best_score = 1e10
    
    np.random.seed(42)
    
    # Try both initializations
    inits = [gl_nodes, cheb_nodes]
    for init_nodes in inits:
        if _time_left() < 15.0:
            break
        params_init = _encode_initial(init_nodes)
        score = _objective_nodes(params_init)
        if score < best_score:
            best_score = score
            best_params = params_init.copy()
    
    if best_params is None:
        best_params = _encode_initial(gl_nodes)
        best_score = _objective_nodes(best_params)
    
    # Nelder-Mead restarts from best and random perturbations
    restarts = 4
    for i in range(restarts):
        if _time_left() < 12.0:
            break
        if i == 0:
            init_p = best_params.copy()
        else:
            init_p = best_params + np.random.randn(8) * 0.4
        try:
            res = minimize(_objective_nodes, init_p, method='Nelder-Mead',
                          options={'maxiter': 1200, 'xatol': 1e-9, 'fatol': 1e-11})
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
    if _time_left() > 18.0:
        test_cosmos_init = [
            (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
            (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
            (-0.2, 0.2), (0.8, -0.8)
        ]
        test_zs_init = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        F_init, T_init = _compute_test_data(nodes, test_cosmos_init, test_zs_init)
        
        # Get initial weights from minimax
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
        
        w_enc = np.log(w_ls + 1e-15)
        joint_init = np.concatenate([best_params, w_enc])
        
        try:
            res_joint = minimize(_joint_objective, joint_init, method='Nelder-Mead',
                                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-13})
            if res_joint.fun < best_score:
                best_params = res_joint.x[:8]
                nodes_raw = 1.0 / (1.0 + np.exp(-best_params))
                nodes = np.sort(nodes_raw)
                nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
                for i in range(1, 8):
                    if nodes[i] <= nodes[i-1]:
                        nodes[i] = nodes[i-1] + 1e-6
        except Exception:
            pass

    # Final weight computation with high-precision minimax
    final_test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1),
        (0.2, -0.5), (-0.8, 0.2)
    ]
    final_test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    F_final, T_final = _compute_test_data(nodes, final_test_cosmos, final_test_zs)
    
    # Use minimax weight solver for final weights
    weights = None
    try:
        # First try the smooth minimax
        W_fin = 1.0 / T_final
        W_fin[W_fin < 1e10] = 0
        Fw_fin = F_final * W_fin[:, None]
        Tw_fin = T_final * W_fin
        
        def obj_fin(w):
            r = Fw_fin @ w - Tw_fin
            abs_r = np.abs(r)
            max_r = np.max(abs_r)
            if max_r < 1e-15:
                return 0.0
            scaled = 30.0 * abs_r / max_r
            return np.log(np.sum(np.exp(scaled))) / 30.0 * max_r
        
        def grad_fin(w):
            r = Fw_fin @ w - Tw_fin
            abs_r = np.abs(r)
            max_r = np.max(abs_r)
            if max_r < 1e-15:
                return np.zeros(8)
            scaled = 30.0 * abs_r / max_r
            exp_scaled = np.exp(scaled)
            sum_exp = np.sum(exp_scaled)
            signs = np.sign(r)
            weights_vec = exp_scaled / sum_exp
            return Fw_fin.T @ (weights_vec * signs)
        
        cons_fin = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
        bounds_fin = [(0, None)] * 8
        w0_fin = np.ones(8) / 8.0
        res_fin = minimize(obj_fin, w0_fin, method='SLSQP', jac=grad_fin, bounds=bounds_fin, constraints=cons_fin,
                          options={'maxiter': 1500, 'ftol': 1e-16})
        if res_fin.success:
            weights = np.maximum(res_fin.x, 0)
            weights = weights / np.sum(weights)
    except Exception:
        pass
    
    if weights is None:
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
