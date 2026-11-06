import numpy as np
from scipy.optimize import linprog
from scipy.integrate import quad
import time

# Constants for cosmology
OM = 0.3
OR = 1e-4
ODE = 1.0 - OM - OR

def E_of_a(a, w0, w1):
    """
    Compute E(a) = H(a)/H0 for CPL dark energy model.
    a: scale factor
    w0, w1: CPL parameters
    """
    if a < 1e-10:
        return np.inf
    
    log_term = np.log(a)
    exp_arg = 3.0 * w1 * (a - 1.0 - log_term)
    
    if exp_arg > 700:
        exp_val = np.exp(700)
    else:
        exp_val = np.exp(exp_arg)
        
    power_term = a ** (-3.0 * (1.0 + w0))
    
    de_term = ODE * power_term * exp_val
    m_term = OM * a ** (-3.0)
    r_term = OR * a ** (-4.0)
    
    sum_term = m_term + r_term + de_term
    
    if sum_term < 0:
        return 0.0
        
    return np.sqrt(sum_term)

def exact_dL(z, w0, w1):
    """
    Compute exact dimensionless luminosity distance.
    """
    if z <= 0:
        return 0.0
        
    def integrand(z_prime):
        a_prime = 1.0 / (1.0 + z_prime)
        e_val = E_of_a(a_prime, w0, w1)
        if e_val == 0 or np.isinf(e_val):
            return 0.0
        return 1.0 / e_val
        
    try:
        val, err = quad(integrand, 0, z, limit=200)
        return (1.0 + z) * val
    except Exception:
        return 0.0

def quad_dL(z, nodes, weights, w0, w1):
    """
    Compute approximate dimensionless luminosity distance using the quadrature rule.
    Formula: (1+z) * z * sum_i weights[i] / E(a_i)
    where a_i = 1 / (1 + z * nodes[i])
    """
    if z <= 0:
        return 0.0
        
    a_vals = 1.0 / (1.0 + z * nodes)
    e_vals = E_of_a(a_vals, w0, w1)
    
    if np.any(e_vals <= 0) or np.any(np.isinf(e_vals)):
        return 0.0
        
    sum_term = np.sum(weights / e_vals)
    return (1.0 + z) * z * sum_term

def fit_minimax_weights(nodes, test_data):
    """
    For fixed nodes, find weights that minimize the worst-case relative error.
    We solve a linear programming problem to minimize t subject to:
      -t * exact_i <= (1+z_i)*z_i*sum_j w_j/E(a_{ij}) - exact_i <= t * exact_i
      sum_j w_j = 1
      w_j >= 0
    """
    n_nodes = len(nodes)
    n_cases = len(test_data)
    
    # Variables: w_0, ..., w_7, t  (9 variables)
    # Objective: minimize t -> c = [0, ..., 0, 1]
    c = np.zeros(n_nodes + 1)
    c[-1] = 1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each case i:
    #   (1+z_i)*z_i*sum_j w_j/E_{ij} - exact_i <= t * exact_i
    #   -> (1+z_i)*z_i*sum_j w_j/E_{ij} - t * exact_i <= exact_i
    #   -> sum_j [(1+z_i)*z_i/E_{ij}] w_j + [-exact_i] t <= exact_i
    #
    #   exact_i - (1+z_i)*z_i*sum_j w_j/E_{ij} <= t * exact_i
    #   -> -sum_j [(1+z_i)*z_i/E_{ij}] w_j + [-exact_i] t <= -exact_i
    #
    # So for each case i, we have two rows.
    
    A_ub = np.zeros((2 * n_cases, n_nodes + 1))
    b_ub = np.zeros(2 * n_cases)
    
    valid_cases = []
    row_idx = 0
    
    for idx, (z, w0, w1) in enumerate(test_data):
        if z <= 0:
            continue
        exact = exact_dL(z, w0, w1)
        if exact <= 0:
            continue
            
        a_vals = 1.0 / (1.0 + z * nodes)
        e_vals = E_of_a(a_vals, w0, w1)
        
        if np.any(e_vals <= 0) or np.any(np.isinf(e_vals)):
            continue
            
        coeff = (1.0 + z) * z / e_vals
        
        # Row 1: sum_j coeff_j w_j - exact_i * t <= exact_i
        A_ub[row_idx, :n_nodes] = coeff
        A_ub[row_idx, n_nodes] = -exact
        b_ub[row_idx] = exact
        row_idx += 1
        
        # Row 2: -sum_j coeff_j w_j - exact_i * t <= -exact_i
        A_ub[row_idx, :n_nodes] = -coeff
        A_ub[row_idx, n_nodes] = -exact
        b_ub[row_idx] = -exact
        row_idx += 1
        
        valid_cases.append(idx)
    
    if row_idx == 0:
        return np.full(n_nodes, 1.0 / n_nodes)
    
    A_ub = A_ub[:row_idx, :]
    b_ub = b_ub[:row_idx]
    
    # Equality constraints: sum w_j = 1
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_j >= 0, t >= 0
    bounds = [(0, None)] * n_nodes + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            w = res.x[:n_nodes]
            # Ensure positive and normalize
            w = np.clip(w, 1e-15, None)
            w = w / np.sum(w)
            return w
        else:
            # Fallback to least squares
            return _lsq_weights(nodes, test_data)
    except Exception:
        return _lsq_weights(nodes, test_data)

def _lsq_weights(nodes, test_data):
    """Fallback: least-squares weights."""
    n_nodes = len(nodes)
    A = []
    y = []
    
    for z, w0, w1 in test_data:
        if z <= 0:
            continue
        exact = exact_dL(z, w0, w1)
        if exact <= 0:
            continue
        a_vals = 1.0 / (1.0 + z * nodes)
        e_vals = E_of_a(a_vals, w0, w1)
        if np.any(e_vals <= 0) or np.any(np.isinf(e_vals)):
            continue
        A.append(1.0 / e_vals)
        y.append(exact / ((1.0 + z) * z))
    
    if len(A) == 0:
        return np.full(n_nodes, 1.0 / n_nodes)
    
    A = np.array(A)
    y = np.array(y)
    
    try:
        w_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        w = np.clip(w_ls, 1e-12, None)
        w = w / np.sum(w)
        return w
    except Exception:
        return np.full(n_nodes, 1.0 / n_nodes)

def worst_relative_error(nodes, weights, test_data):
    """Compute the worst-case relative error in percent."""
    max_err = 0.0
    for z, w0, w1 in test_data:
        if z <= 0:
            continue
        exact = exact_dL(z, w0, w1)
        if exact <= 0:
            continue
        approx = quad_dL(z, nodes, weights, w0, w1)
        rel_err = abs((approx - exact) / exact)
        if rel_err > max_err:
            max_err = rel_err
    return max_err * 100.0

def generate_node_families():
    """Generate diverse node families for 8 nodes in (0,1)."""
    families = []
    
    # 1. Midpoint rule
    n = 8
    families.append((np.arange(n) + 0.5) / n)
    
    # 2. Gauss-Legendre nodes (8-point)
    try:
        nodes_gl, _ = np.polynomial.legendre.leggauss(8)
        # Map from [-1, 1] to [0, 1]
        nodes_gl = (nodes_gl + 1.0) / 2.0
        families.append(nodes_gl)
    except Exception:
        pass
    
    # 3. Beta(2,2) quantiles
    from scipy.stats import beta as beta_dist
    qs = np.linspace(0, 1, 9)
    nodes_beta = beta_dist.ppf(qs[1:-1], 2, 2)
    families.append(nodes_beta)
    
    # 4. Beta(1,3) quantiles (skewed toward 0)
    nodes_beta13 = beta_dist.ppf(qs[1:-1], 1, 3)
    families.append(nodes_beta13)
    
    # 5. Beta(3,1) quantiles (skewed toward 1)
    nodes_beta31 = beta_dist.ppf(qs[1:-1], 3, 1)
    families.append(nodes_beta31)
    
    # 6. Log-spaced
    nodes_log = np.geomspace(0.01, 0.99, 8)
    families.append(nodes_log)
    
    # 7. Power-law: x^0.5
    nodes_pow = np.linspace(0.01, 0.99, 8) ** 0.5
    nodes_pow = np.sort(nodes_pow)
    families.append(nodes_pow)
    
    # 8. Power-law: x^2
    nodes_pow2 = np.linspace(0.01, 0.99, 8) ** 2
    nodes_pow2 = np.sort(nodes_pow2)
    families.append(nodes_pow2)
    
    # 9. Sine-spaced: sin(pi*x/2)
    x_lin = np.linspace(0.01, 0.99, 8)
    nodes_sine = np.sin(np.pi * x_lin / 2.0)
    nodes_sine = np.sort(nodes_sine)
    families.append(nodes_sine)
    
    # 10. Cosine-spaced
    nodes_cos = np.cos(np.linspace(0, np.pi, 8))
    nodes_cos = (nodes_cos + 1.0) / 2.0
    nodes_cos = np.sort(nodes_cos)
    families.append(nodes_cos)
    
    # 11. Uniform with jitter (deterministic)
    rng = np.random.RandomState(42)
    nodes_jitter = np.sort(rng.uniform(0.01, 0.99, 8))
    families.append(nodes_jitter)
    
    # 12. Truncated Gauss-Hermite style
    nodes_gh = np.array([0.05, 0.15, 0.25, 0.38, 0.52, 0.65, 0.80, 0.95])
    families.append(nodes_gh)
    
    # 13. Emphasize low-z (more nodes near 0)
    nodes_lowz = np.array([0.02, 0.06, 0.12, 0.20, 0.35, 0.55, 0.75, 0.95])
    families.append(nodes_lowz)
    
    # 14. Emphasize high-z
    nodes_highz = np.array([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 0.97])
    families.append(nodes_highz)
    
    # 15. Chebyshev nodes mapped to [0,1]
    k = np.arange(8)
    nodes_cheb = 0.5 * (1 + np.cos(np.pi * (2*k + 1) / 16))
    nodes_cheb = np.sort(nodes_cheb)
    families.append(nodes_cheb)
    
    return families

def local_node_refinement(nodes, weights, test_data, max_iter=30):
    """
    Simple local refinement: perturb nodes slightly and refit weights.
    Use coordinate descent or random perturbations.
    """
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = worst_relative_error(best_nodes, best_weights, test_data)
    
    rng = np.random.RandomState(123)
    
    for it in range(max_iter):
        # Perturb nodes
        perturbation = rng.normal(0, 0.02, size=8)
        new_nodes = best_nodes + perturbation
        new_nodes = np.clip(new_nodes, 1e-4, 1 - 1e-4)
        new_nodes = np.sort(new_nodes)
        
        # Ensure strictly increasing
        if not np.all(np.diff(new_nodes) > 1e-6):
            continue
            
        # Refit weights
        new_weights = fit_minimax_weights(new_nodes, test_data)
        new_err = worst_relative_error(new_nodes, new_weights, test_data)
        
        if new_err < best_err:
            best_nodes = new_nodes
            best_weights = new_weights
            best_err = new_err
    
    return best_nodes, best_weights, best_err

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s
    
    # Define test data: (z, w0, w1)
    # Cover a good range of behavior
    zs = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    w0s = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.3]
    w1s = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    test_data = []
    for z in zs:
        for w0 in w0s:
            for w1 in w1s:
                test_data.append((z, w0, w1))
    
    # To keep computation fast, use a subset for optimization
    # Full set for final verification
    n_total = len(test_data)
    # Use all if not too many
    if n_total <= 150:
        opt_data = test_data
    else:
        # Subsample
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total, size=min(100, n_total), replace=False)
        opt_data = [test_data[i] for i in indices]
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Generate diverse node families
    families = generate_node_families()
    
    for family in families:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        nodes = np.sort(family)
        # Ensure in (0,1) and strictly increasing
        nodes = np.clip(nodes, 1e-4, 1 - 1e-4)
        if not np.all(np.diff(nodes) > 1e-6):
            continue
        
        # Fit optimal weights
        weights = fit_minimax_weights(nodes, opt_data)
        
        # Evaluate
        err = worst_relative_error(nodes, weights, opt_data)
        
        if err < best_err:
            best_err = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    # Local refinement on the best node set
    if best_nodes is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            best_nodes, best_weights, best_err = local_node_refinement(
                best_nodes, best_weights, opt_data, max_iter=20
            )
    
    # Final verification on full test data
    if best_nodes is not None:
        final_err = worst_relative_error(best_nodes, best_weights, test_data)
        
        # If the error on full data is significantly worse, refit weights on full data
        if final_err > best_err * 1.1:
            best_weights = fit_minimax_weights(best_nodes, test_data)
            final_err = worst_relative_error(best_nodes, best_weights, test_data)
    
    # Final safety checks
    if best_nodes is None or best_weights is None:
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Ensure nodes are strictly increasing in [0,1]
    if not np.all(np.diff(best_nodes) > 1e-12):
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    if best_nodes[0] < 0 or best_nodes[-1] > 1:
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Ensure weights are positive and sum to 1
    best_weights = np.clip(best_weights, 1e-15, None)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
