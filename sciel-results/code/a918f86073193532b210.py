<<<<<<< SEARCH
def _stochastic_refine(params, deadline, rng, n_iters=6000):
    """Simulated annealing style local search."""
    val = _penalty_objective(params, penalty=5000.0)
    best = params.copy()
    best_val = val
    T = 0.03
    step = 0.015

    for it in range(n_iters):
        if time.time() > deadline:
            break
        i = rng.randint(N)
        cand = params.copy()
        cand[3*i] = params[3*i] + rng.uniform(-step, step)
        cand[3*i+1] = params[3*i+1] + rng.uniform(-step, step)
        cand[3*i+2] = params[3*i+2] + rng.uniform(-step*0.4, step*0.4)
        cand[3*i] = np.clip(cand[3*i], 0.0, 1.0)
        cand[3*i+1] = np.clip(cand[3*i+1], 0.0, 1.0)
        cand[3*i+2] = np.clip(cand[3*i+2], 1e-6, 0.5)

        new_val = _penalty_objective(cand, penalty=5000.0)
        d = new_val - val
        if d < 0 or rng.rand() < np.exp(-d / max(T, 1e-8)):
            params = cand
            val = new_val
            if val < best_val:
                best_val = val
                best = params.copy()
        T *= 0.9992
        step *= 0.9997

    return best, best_val
=======
def _stochastic_refine(params, deadline, rng, n_iters=6000):
    """Simulated annealing style local search."""
    val = _penalty_objective(params, penalty=5000.0)
    best = params.copy()
    best_val = val
    T = 0.03
    step = 0.015

    for it in range(n_iters):
        if time.time() > deadline:
            break
        i = rng.randint(N)
        cand = params.copy()
        cand[3*i] = params[3*i] + rng.uniform(-step, step)
        cand[3*i+1] = params[3*i+1] + rng.uniform(-step, step)
        cand[3*i+2] = params[3*i+2] + rng.uniform(-step*0.4, step*0.4)
        cand[3*i] = np.clip(cand[3*i], 0.0, 1.0)
        cand[3*i+1] = np.clip(cand[3*i+1], 0.0, 1.0)
        cand[3*i+2] = np.clip(cand[3*i+2], 1e-6, 0.5)

        new_val = _penalty_objective(cand, penalty=5000.0)
        d = new_val - val
        if d < 0 or rng.rand() < np.exp(-d / max(T, 1e-8)):
            params = cand
            val = new_val
            if val < best_val:
                best_val = val
                best = params.copy()
        T *= 0.9992
        step *= 0.9997

    return best, best_val


def _slsqp_optimize(params, deadline, max_time=10.0):
    """SLSQP optimization using Delaunay-edge constraints."""
    p = params.reshape(N, 3).copy()
    r = np.maximum(p[:, 2], 1e-6)
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    
    # Build constraints
    cons = _slsqp_constraints(params)
    
    try:
        res = minimize(
            _slsqp_objective, params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
        )
        if res.success or res.fun < _slsqp_objective(params):
            return res.x
    except Exception:
        pass
    
    return params
>>>>>>> REPLACE

