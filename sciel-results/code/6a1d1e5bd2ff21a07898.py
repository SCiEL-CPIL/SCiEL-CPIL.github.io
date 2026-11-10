def _solve_lp_reduced(K):
    max_x = 10 * K
    # Constraints: M(i*K) <= 1 for i=1..10
    x_checks = np.arange(1, 11) * K  # [K, 2K, ..., 10K]
    n_cons = len(x_checks)
    
    # A_ub: n_cons x K
    A_ub = np.zeros((n_cons, K), dtype=np.float64)
    for i, x in enumerate(x_checks):
        for j in range(K):
            A_ub[i, j] = x // (j + 1)
    
    b_ub = np.ones(n_cons)
    
    c = np.log(np.arange(1, K + 1)) / np.arange(1, K + 1)
    
    A_eq = (1.0 / np.arange(1, K + 1)).reshape(1, -1)
    b_eq = np.array([0.0])
    
    bounds = [(-10.0, 10.0)] * K
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return None
    
    return res.x
