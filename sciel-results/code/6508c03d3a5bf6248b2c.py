<<<<<<< SEARCH
def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0
=======
def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective with epigraph trick:
    minimize t s.t. C_k(h) <= t for all k, sum(h)=N/2, 0<=h<=1.
    This gives a smoother optimization landscape.
    """
    n = N
    # State: [h (n vars), t (1 var)]
    x0 = np.concatenate([h0, [0.5]])
    
    bounds = [(0.0, 1.0)] * n + [(-1.0, 2.0)]
    
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - N / 2.0,
             'jac': lambda x: np.concatenate([np.ones(n), [0.0]])}]
    
    # Inequality constraints: C_k(h) - t <= 0 for each k
    # We use a subset of the most critical constraints to keep it manageable
    def _get_critical_constraints():
        """Get indices of top overlap values for constraint selection."""
        cs = _linear_overlap_values(h0)
        # Use top 20 constraints
        top_k = np.argsort(cs)[-20:]
        return sorted(top_k.tolist())
    
    critical_k = _get_critical_constraints()
    
    for k in critical_k:
        def _ineq(x, k=k):
            h = x[:n]
            cs = _linear_overlap_values(h)
            return x[n] - cs[k]
        
        def _ineq_jac(x, k=k):
            h = x[:n]
            _, grad_h = _overlap_and_grad(h, beta)
            # d(C_k)/d(h) contribution
            # grad_h already includes all C_k weighted; we need just C_k's gradient
            # Recompute for single k
            n_loc = len(h)
            dCk = np.zeros(n_loc)
            factor = -(2.0 / n_loc)
            if k > 0:
                dCk[k:] += factor * h[:n_loc - k]
            dCk[:n_loc - k] += factor * h[k:]
            return np.concatenate([dCk, [-1.0]])
        
        cons.append({'type': 'ineq', 'fun': _ineq, 'jac': _ineq_jac})
    
    def obj_func(x):
        return x[n]  # minimize t
    
    def jac_func(x):
        return np.concatenate([np.zeros(n), [1.0]])
    
    try:
        res = minimize(obj_func, x0, jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if res.success or res.fun < x0[n]:
            h_opt = res.x[:n]
            h_opt = _fix_sum(h_opt)
            return h_opt
    except Exception:
        pass
    
    # Fallback to original SLSQP on smooth max
    cons2 = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
              'jac': lambda x: np.ones(N)}]
    bounds2 = [(0.0, 1.0)] * N
    
    def obj_func2(x):
        val, _ = _overlap_and_grad(x, beta)
        return val
    
    def jac_func2(x):
        _, grad = _overlap_and_grad(x, beta)
        return grad
    
    try:
        res2 = minimize(obj_func2, h0.copy(), jac=jac_func2, method='SLSQP',
                        bounds=bounds2, constraints=cons2,
                        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res2.x
    except Exception:
        return h0
>>>>>>> REPLACE

