<<<<<<< SEARCH
def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Delaunay-based with moderate penalty
    penalty = 10.0
    smooth_eps = 1e-4
    remaining = deadline - time.time()
    if remaining > 3:
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_delaunay(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Increase penalty, full pairwise
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 50.0
        smooth_eps = 1e-5
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Very high penalty for fine-tuning
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 200.0
        smooth_eps = 1e-6
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Final: even higher penalty
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 1000.0
        smooth_eps = 1e-7
        try:
            res = minimize(_objective_vectorized, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_vectorized(params, penalty, smooth_eps):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score
=======
def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty and k-NN."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Phase 1: k-NN based, moderate penalty
    remaining = _remaining(deadline)
    if remaining > 4:
        penalty = 5.0
        smooth_eps = 1e-4
        k = 12
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps, k),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            cur_obj = _objective_knn(params, penalty, smooth_eps, k)
            if res.fun < cur_obj:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: k-NN with higher penalty
    remaining = _remaining(deadline)
    if remaining > 4:
        penalty = 30.0
        smooth_eps = 1e-5
        k = 12
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps, k),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            cur_obj = _objective_knn(params, penalty, smooth_eps, k)
            if res.fun < cur_obj:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: Full pairwise, high penalty
    remaining = _remaining(deadline)
    if remaining > 3:
        penalty = 100.0
        smooth_eps = 1e-6
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            cur_obj = _objective_full(params, penalty, smooth_eps)
            if res.fun < cur_obj:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Full pairwise, very high penalty
    remaining = _remaining(deadline)
    if remaining > 2.5:
        penalty = 500.0
        smooth_eps = 1e-7
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            cur_obj = _objective_full(params, penalty, smooth_eps)
            if res.fun < cur_obj:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Final polish with extreme penalty
    remaining = _remaining(deadline)
    if remaining > 2.0:
        penalty = 2000.0
        smooth_eps = 1e-8
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            cur_obj = _objective_full(params, penalty, smooth_eps)
            if res.fun < cur_obj:
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score
>>>>>>> REPLACE

