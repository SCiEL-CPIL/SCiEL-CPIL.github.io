<<<<<<< SEARCH
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
=======
def _joint_objective(params):
    """Joint optimization of nodes and weights using epigraph formulation with smooth surrogate."""
    if _time_left() < 2.0:
        return 1e10
    # First 8 params: node locations (sigmoid encoded)
    # Next 8 params: weights (log-encoded for positivity)
    # Last param: t (epigraph variable for max relative error)
    node_params = params[:8]
    weight_params = params[8:16]
    t = params[16]
    
    nodes = _decode_params(node_params)
    
    # Log-softmax weights for positivity
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
    
    # Epigraph: minimize t + penalty for violations
    # t >= rel_errors[i] for all i, smooth via log-sum-exp
    alpha = 100.0
    lse = _log_sum_exp(rel_errors, alpha)
    
    # Penalty if t < lse (smooth hinge)
    violation = np.maximum(0.0, lse - t)
    penalty = 10.0 * violation**2
    
    return t + penalty
>>>>>>> REPLACE

