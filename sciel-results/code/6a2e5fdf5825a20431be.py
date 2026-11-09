<<<<<<< SEARCH
def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 8 node params (t), 8 weight params (v).
    """
    if _time_left() < 1.0:
        return 1e10
        
    # Decode nodes
    # t_i -> x_i sequence
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        # Sigmoid mapping to (0, 1)
        s = 1.0 / (1.0 + np.exp(-params[i]))
        # Scale step by remaining space
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    # Ensure strictly increasing and in (0,1)
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    # Decode weights
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Test set
    # We need a representative set of cosmologies.
    # Include LCDM, wCDM, and some extreme CPL cases.
    # The evaluator uses "disjoint public validation cosmologies".
    # We should sample a broad range of w0, wa.
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   
        (0.3, -0.3),   
        (-2.0, 1.0),   # More extreme phantom
        (1.0, -2.0)    # More extreme quintessence
    ]
    
    # Redshifts
    # Cover a wide range
    # The evaluator likely tests a specific set. We should cover 0 < z < maybe 10 or 20.
    # Low z is critical because relative error can be amplified if dL is small?
    # dL ~ z for small z.
    test_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, x, w, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err
=======
def _compute_moment_matrix(nodes, test_cosmos, test_zs):
    """
    For given nodes, compute the matrix M where M[i,j] is the integral of the j-th
    basis function evaluated at node i across test cases.
    Actually, we want to find weights that minimize worst-case error.
    
    For a fixed set of nodes, the quadrature approximation for each (cosmo, z) pair is:
      Q_j = sum_i w_i * f_j(x_i)  where f_j is the integrand for case j
    True value: T_j = integral of f_j
    
    We want to minimize max_j |Q_j - T_j| / T_j subject to sum(w_i) = 1, w_i > 0.
    
    This is a linear programming problem if we normalize by T_j:
      minimize t
      subject to: -t <= sum_i (w_i * f_j(x_i) / T_j) - 1 <= t  for all j
                  sum_i w_i = 1
                  w_i >= 0
    """
    n_cases = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            n_cases += 1
    
    # Build A matrix: A[j, i] = f_j(x_i) / T_j
    # And b vector: b[j] = 1.0
    A = np.zeros((n_cases, 8))
    case_idx = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            if true_val <= 1e-10:
                A[case_idx, :] = 0.0
            else:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa)
                # f_j(x_i) = 1/E(a_i) * z * (1+z) ... wait, let me re-derive
                # dL = (1+z) * integral_0^z dζ / E(1/(1+ζ))
                # Quadrature: dL ≈ (1+z) * z * sum_i w_i / E(1/(1+z*x_i))
                # So Q_j = (1+z) * z * sum_i w_i / E(1/(1+z*x_i))
                # T_j = true dL
                # Q_j / T_j = [(1+z) * z * sum_i w_i / E(1/(1+z*x_i))] / T_j
                # We want Q_j / T_j ≈ 1
                # So: sum_i [w_i * (1+z) * z / E(1/(1+z*x_i))] / T_j ≈ 1
                # A[j, i] = (1+z) * z / (E(1/(1+z*x_i)) * T_j)
                A[case_idx, :] = (1.0 + z) * z / (E_vals * true_val)
            case_idx += 1
    
    return A

def _solve_weights_lp(nodes, test_cosmos, test_zs):
    """
    Solve the LP to find optimal weights for given nodes.
    Returns weights or None if infeasible.
    """
    A = _compute_moment_matrix(nodes, test_cosmos, test_zs)
    n_cases, n_nodes = A.shape
    
    # LP: minimize t
    # s.t. -t <= A @ w - 1 <= t
    #      sum(w) = 1
    #      w >= 0
    #
    # Variables: [w_0, ..., w_7, t]  (9 variables)
    #
    # Constraints:
    # A @ w - t <= 1   (n_cases constraints)
    # -A @ w - t <= -1  (n_cases constraints)
    # sum(w) = 1       (1 equality)
    # w >= 0           (handled by bounds)
    # t >= 0           (handled by bounds)
    
    c = np.zeros(9)
    c[8] = 1.0  # minimize t
    
    # Inequality constraints: A_ub @ x <= b_ub
    # A @ w - t <= 1  =>  [A, -I] @ [w, t] <= 1
    # -A @ w - t <= -1 =>  [-A, -I] @ [w, t] <= -1
    A_ub = np.zeros((2 * n_cases, 9))
    b_ub = np.zeros(2 * n_cases)
    
    A_ub[:n_cases, :8] = A
    A_ub[:n_cases, 8] = -1.0
    b_ub[:n_cases] = 1.0
    
    A_ub[n_cases:, :8] = -A
    A_ub[n_cases:, 8] = -1.0
    b_ub[n_cases:] = -1.0
    
    # Equality constraint: sum(w) = 1
    A_eq = np.zeros((1, 9))
    A_eq[0, :8] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 0, t >= 0
    bounds = [(0, None)] * 8 + [(0, None)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            w = result.x[:8]
            # Ensure numerical positivity
            w = np.maximum(w, 1e-15)
            w = w / np.sum(w)
            return w
    except Exception:
        pass
    
    return None

def _worst_case_error(nodes, weights, test_cosmos, test_zs):
    """Compute worst-case relative error for given nodes and weights."""
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _node_objective(node_params, test_cosmos, test_zs):
    """
    Objective for node optimization: given node parameters, find optimal weights
    via LP, then return worst-case error.
    """
    if _time_left() < 2.0:
        return 1e10
    
    # Decode nodes from parameters using sequential sigmoid
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        s = 1.0 / (1.0 + np.exp(-node_params[i]))
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    # Ensure strictly increasing
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-12
    
    # Solve LP for optimal weights
    w = _solve_weights_lp(x, test_cosmos, test_zs)
    if w is None:
        # Fallback: use Gauss-Legendre weights as starting point
        _, w_gl = np.polynomial.legendre.leggauss(8)
        w = 0.5 * w_gl
        w = w / np.sum(w)
    
    return _worst_case_error(x, w, test_cosmos, test_zs)
>>>>>>> REPLACE

