"""Chebyshev equioscillation approach for luminosity distance quadrature."""
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

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _rel_error_vector(nodes, weights, test_cosmos, test_zs):
    """Compute vector of relative errors."""
    errors = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
    return np.array(errors)

def _solve_weights_lp(nodes, test_cosmos, test_zs, max_iter=50):
    """
    Solve for weights using LP to minimize worst-case relative error.
    Variables: w_1, ..., w_8, epsilon
    Minimize: epsilon
    Subject to:
      -epsilon <= (quad_val - true_val)/true_val <= epsilon  for each (cosmo, z)
      sum(w_i) = 1
      w_i >= 0
    """
    n_cosmos = len(test_cosmos)
    n_zs = len(test_zs)
    n_constraints = n_cosmos * n_zs
    n_vars = 9  # 8 weights + epsilon
    
    # Build constraint matrix
    # For each (cosmo_idx, z_idx):
    #   sum_i w_i / E(a_i) * z * (1+z) - true_val <= true_val * epsilon
    #   -(sum_i w_i / E(a_i) * z * (1+z) - true_val) <= true_val * epsilon
    #   => sum_i w_i / E(a_i) * z * (1+z) - true_val*epsilon <= true_val
    #   => -sum_i w_i / E(a_i) * z * (1+z) + true_val*epsilon <= true_val
    
    A_ub = np.zeros((2 * n_constraints, n_vars))
    b_ub = np.zeros(2 * n_constraints)
    
    idx = 0
    for ci, (w0, wa) in enumerate(test_cosmos):
        for zi, z in enumerate(test_zs):
            true_val = _true_dL(z, w0, wa)
            if true_val < 1e-10:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            coeffs = (z * (1.0 + z) / E_vals)  # 8 values
            
            # Constraint 1: sum(w_i * coeffs_i) - true_val*eps <= true_val
            A_ub[idx, :8] = coeffs
            A_ub[idx, 8] = -true_val
            b_ub[idx] = true_val
            idx += 1
            
            # Constraint 2: -sum(w_i * coeffs_i) + true_val*eps <= true_val
            A_ub[idx, :8] = -coeffs
            A_ub[idx, 8] = true_val
            b_ub[idx] = true_val
            idx += 1
    
    # Trim unused rows
    A_ub = A_ub[:idx, :]
    b_ub = b_ub[:idx]
    
    # Equality constraint: sum(w_i) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :8] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 0, epsilon >= 0
    bounds = [(0, None)] * 8 + [(0, None)]
    
    try:
        result = linprog(c=np.zeros(n_vars) + np.array([0]*8 + [1]),
                        A_ub=A_ub, b_ub=b_ub,
                        A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        if result.success:
            weights = result.x[:8]
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            return weights, result.x[8]
        else:
            # Fallback: equal weights
            return np.ones(8) / 8.0, np.inf
    except Exception:
        return np.ones(8) / 8.0, np.inf

def _solve_weights_ls(nodes, test_cosmos, test_zs):
    """
    Solve for weights using least squares to minimize sum of squared relative errors.
    """
    n_cosmos = len(test_cosmos)
    n_zs = len(test_zs)
    n_constraints = n_cosmos * n_zs
    
    # Build system: A @ w = b where A[i] = coeffs_i, b[i] = true_val_i
    A = np.zeros((n_constraints, 8))
    b = np.zeros(n_constraints)
    
    idx = 0
    for ci, (w0, wa) in enumerate(test_cosmos):
        for zi, z in enumerate(test_zs):
            true_val = _true_dL(z, w0, wa)
            if true_val < 1e-10:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            coeffs = (z * (1.0 + z) / E_vals)
            A[idx, :] = coeffs
            b[idx] = true_val
            idx += 1
    
    A = A[:idx, :]
    b = b[:idx]
    
    try:
        # Solve least squares with non-negativity constraint
        from scipy.optimize import nnls
        w, residual = nnls(A, b)
        w_sum = np.sum(w)
        if w_sum < 1e-15:
            return np.ones(8) / 8.0
        w = w / w_sum
        return w
    except Exception:
        return np.ones(8) / 8.0

def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _chebyshev_nodes_init():
    """
    Initialize nodes using Chebyshev roots mapped to [0,1].
    Chebyshev nodes cluster near the endpoints, which may be beneficial
    for the non-uniform behavior of 1/E(a) near a=1 (z=0) and a->0 (z->inf).
    """
    n = 8
    k = np.arange(n)
    # Chebyshev roots in [-1, 1]
    x = np.cos((2*k + 1) * np.pi / (2*n))
    # Map to [0, 1]
    nodes = 0.5 * (x + 1.0)
    nodes = np.sort(nodes)
    
    # Compute initial weights using Gauss-Chebyshev quadrature weights
    # For Chebyshev nodes, the weights are pi/n for all nodes (for the standard integral)
    # But we need to adjust for the mapping and the specific integrand
    weights = np.ones(n) / n
    
    return nodes, weights

def _remez_iteration(nodes, weights, test_cosmos, test_zs, max_iter=30):
    """
    Remez-like algorithm to find the minimax quadrature rule.
    Alternates between:
    1. Given nodes, solve for optimal weights (LP or LS)
    2. Given weights, adjust nodes to equalize errors
    """
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_max_err = np.inf
    
    for iteration in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Step 1: Solve for optimal weights given current nodes
        opt_weights, lp_err = _solve_weights_lp(nodes, test_cosmos, test_zs)
        
        # Evaluate current max error
        errors = _rel_error_vector(nodes, opt_weights, test_cosmos, test_zs)
        max_err = np.max(errors)
        
        if max_err < best_max_err:
            best_max_err = max_err
            best_nodes = nodes.copy()
            best_weights = opt_weights.copy()
        
        # Step 2: Adjust nodes using gradient-based optimization
        # We want to minimize the max error by moving nodes
        def _node_objective(node_params):
            # Decode nodes from parameters
            n_nodes = 8
            raw_nodes = node_params[:n_nodes]
            # Use sigmoid to ensure nodes are in (0, 1) and sorted
            sig_nodes = 1.0 / (1.0 + np.exp(-10.0 * (raw_nodes - 3.5)))
            sig_nodes = np.sort(sig_nodes)
            # Ensure strict ordering
            for i in range(1, n_nodes):
                if sig_nodes[i] <= sig_nodes[i-1] + 1e-12:
                    sig_nodes[i] = sig_nodes[i-1] + 1e-12
            sig_nodes = np.clip(sig_nodes, 1e-10, 1.0 - 1e-10)
            
            # Solve for optimal weights
            w_opt, _ = _solve_weights_lp(sig_nodes, test_cosmos, test_zs)
            
            # Compute max error
            errs = _rel_error_vector(sig_nodes, w_opt, test_cosmos, test_zs)
            return np.max(errs)
        
        # Encode current nodes
        raw_nodes = np.log(nodes / (1.0 - nodes)) / 10.0 + 3.5
        initial_params = raw_nodes
        
        try:
            res = minimize(_node_objective, initial_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_max_err:
                new_raw = res.x[:8]
                new_nodes = 1.0 / (1.0 + np.exp(-10.0 * (new_raw - 3.5)))
                new_nodes = np.sort(new_nodes)
                for i in range(1, 8):
                    if new_nodes[i] <= new_nodes[i-1] + 1e-12:
                        new_nodes[i] = new_nodes[i-1] + 1e-12
                new_nodes = np.clip(new_nodes, 1e-10, 1.0 - 1e-10)
                
                # Solve for optimal weights at new nodes
                new_weights, _ = _solve_weights_lp(new_nodes, test_cosmos, test_zs)
                
                new_errors = _rel_error_vector(new_nodes, new_weights, test_cosmos, test_zs)
                new_max_err = np.max(new_errors)
                
                if new_max_err < best_max_err:
                    nodes = new_nodes
                    weights = new_weights
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
                    best_max_err = new_max_err
        except Exception:
            pass
    
    return best_nodes, best_weights, best_max_err

def _joint_optimize(nodes, weights, test_cosmos, test_zs, max_iter=50):
    """
    Jointly optimize nodes and weights using a combined approach.
    """
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_max_err = np.inf
    
    for iteration in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Solve for optimal weights
        opt_weights, _ = _solve_weights_lp(nodes, test_cosmos, test_zs)
        
        # Evaluate
        errors = _rel_error_vector(nodes, opt_weights, test_cosmos, test_zs)
        max_err = np.max(errors)
        
        if max_err < best_max_err:
            best_max_err = max_err
            best_nodes = nodes.copy()
            best_weights = opt_weights.copy()
        
        # Adjust nodes using a smoothed objective
        def _smoothed_objective(params):
            raw_nodes = params[:8]
            sig_nodes = 1.0 / (1.0 + np.exp(-10.0 * (raw_nodes - 3.5)))
            sig_nodes = np.sort(sig_nodes)
            for i in range(1, 8):
                if sig_nodes[i] <= sig_nodes[i-1] + 1e-12:
                    sig_nodes[i] = sig_nodes[i-1] + 1e-12
            sig_nodes = np.clip(sig_nodes, 1e-10, 1.0 - 1e-10)
            
            w_opt, _ = _solve_weights_lp(sig_nodes, test_cosmos, test_zs)
            
            errs = _rel_error_vector(sig_nodes, w_opt, test_cosmos, test_zs)
            # Use log-sum-exp for smooth max
            max_e = np.max(errs)
            if max_e < 1e-15:
                return 0.0
            log_sum = np.log(np.sum(np.exp(50.0 * (errs - max_e)))) / 50.0 + max_e
            return log_sum
        
        raw_nodes = np.log(nodes / (1.0 - nodes)) / 10.0 + 3.5
        initial_params = raw_nodes
        
        try:
            res = minimize(_smoothed_objective, initial_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_max_err:
                new_raw = res.x[:8]
                new_nodes = 1.0 / (1.0 + np.exp(-10.0 * (new_raw - 3.5)))
                new_nodes = np.sort(new_nodes)
                for i in range(1, 8):
                    if new_nodes[i] <= new_nodes[i-1] + 1e-12:
                        new_nodes[i] = new_nodes[i-1] + 1e-12
                new_nodes = np.clip(new_nodes, 1e-10, 1.0 - 1e-10)
                
                new_weights, _ = _solve_weights_lp(new_nodes, test_cosmos, test_zs)
                
                new_errors = _rel_error_vector(new_nodes, new_weights, test_cosmos, test_zs)
                new_max_err = np.max(new_errors)
                
                if new_max_err < best_max_err:
                    nodes = new_nodes
                    weights = new_weights
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
                    best_max_err = new_max_err
        except Exception:
            pass
    
    return best_nodes, best_weights, best_max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 88.0
    
    np.random.seed(42)
    
    # Test cosmologies for optimization
    test_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.3),
        (-2.0, 1.0),
        (1.0, -2.0),
        (-0.8, 0.3),
        (0.2, -0.5),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Standard Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_init()
    initializations.append((gl_nodes, gl_weights))
    
    # 2. Chebyshev nodes
    ch_nodes, ch_weights = _chebyshev_nodes_init()
    initializations.append((ch_nodes, ch_weights))
    
    # 3. Random perturbations around GL
    for _ in range(10):
        perturbation = np.random.randn(8) * 0.1
        pert_nodes = gl_nodes + perturbation
        pert_nodes = np.sort(pert_nodes)
        pert_nodes = np.clip(pert_nodes, 0.01, 0.99)
        # Ensure strict ordering
        for i in range(1, 8):
            if pert_nodes[i] <= pert_nodes[i-1] + 1e-6:
                pert_nodes[i] = pert_nodes[i-1] + 1e-6
        initializations.append((pert_nodes, gl_weights.copy()))
    
    # 4. Random perturbations around Chebyshev
    for _ in range(10):
        perturbation = np.random.randn(8) * 0.1
        pert_nodes = ch_nodes + perturbation
        pert_nodes = np.sort(pert_nodes)
        pert_nodes = np.clip(pert_nodes, 0.01, 0.99)
        for i in range(1, 8):
            if pert_nodes[i] <= pert_nodes[i-1] + 1e-6:
                pert_nodes[i] = pert_nodes[i-1] + 1e-6
        initializations.append((pert_nodes, ch_weights.copy()))
    
    best_nodes = initializations[0][0].copy()
    best_weights = initializations[0][1].copy()
    best_max_err = np.inf
    
    # Optimize from each initialization
    for init_nodes, init_weights in initializations:
        if _time_left() < 10.0:
            break
        
        try:
            opt_nodes, opt_weights, opt_err = _joint_optimize(
                init_nodes.copy(), init_weights.copy(), 
                test_cosmos, test_zs, max_iter=30
            )
            
            if opt_err < best_max_err:
                best_max_err = opt_err
                best_nodes = opt_nodes.copy()
                best_weights = opt_weights.copy()
        except Exception:
            pass
    
    # Final refinement with more iterations
    if _time_left() > 10.0:
        try:
            opt_nodes, opt_weights, opt_err = _joint_optimize(
                best_nodes.copy(), best_weights.copy(),
                test_cosmos, test_zs, max_iter=50
            )
            if opt_err < best_max_err:
                best_max_err = opt_err
                best_nodes = opt_nodes.copy()
                best_weights = opt_weights.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_weights = np.abs(best_weights)
    if np.sum(best_weights) < 1e-15:
        best_weights = np.ones(8) / 8.0
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
