import numpy as np
import time

N = 100
DEADLINE = 150.0
SAFETY = 5.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _correlation_full(h):
    """Compute full correlation array."""
    return np.correlate(h, 1 - h, mode="full")

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(100):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_objective(h, T):
    """
    Smooth approximation of max(corr) using log-sum-exp.
    """
    corr = _correlation_full(h)
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    sum_exp = np.sum(np.exp(shifted))
    lse = x_max + np.log(sum_exp)
    return T * lse

def _soft_max_grad(h, T):
    """
    Gradient of the soft-max objective w.r.t. h.
    """
    corr = _correlation_full(h)
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # grad of corr_k w.r.t. h_i
    # corr_k = sum_j h_j * (1 - h_{j+k}) for k >= 0
    # For k < 0: corr_k = sum_j h_{j-k} * (1 - h_j)
    
    grad = np.zeros(N)
    for k in range(-N+1, N):
        w = weights[k + N - 1]
        if k >= 0:
            # corr_k = sum_{j=0}^{N-1-k} h_j * (1 - h_{j+k})
            for j in range(N - k):
                # d/dh_j [h_j*(1-h_{j+k})] = 1 - h_{j+k}
                # d/dh_{j+k} [h_j*(1-h_{j+k})] = -h_j
                grad[j] += w * (1 - h[j+k])
                if j + k != j:
                    grad[j+k] += w * (-h[j])
        else:
            # k < 0, let m = -k > 0
            # corr_k = sum_{j=0}^{N-1+m} h_{j-m} * (1 - h_j)  ... need to be careful
            # Actually np.correlate(h, 1-h, mode='full')
            # corr_full[k] = sum_i h[i] * (1-h)[i+k-N+1] ... 
            # Let's just use numerical gradient for the soft max for simplicity and correctness
            pass
    
    # Actually, let's just use numerical gradient for the full objective since N=100 is manageable
    eps = 1e-6
    grad_num = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        grad_num[i] = (_soft_max_objective(h_plus, T) - _soft_max_objective(h_minus, T)) / (2 * eps)
    
    return grad_num

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Phase 1: Differential Evolution on low-dimensional space (10 control points)
    # This allows for more effective global exploration than 100-dim DE.
    if time.time() < deadline - 60:
        def _low_dim_to_high_dim(params):
            """Interpolate 10 control points to 100 heights."""
            indices = np.linspace(0, 99, len(params))
            h = np.interp(np.arange(N), indices, params)
            return h

        def de_objective_low_dim(params):
            h = _low_dim_to_high_dim(params)
            h = _project_to_feasible(h)
            return _correlation_value(h)
        
        bounds_low_dim = [(0.0, 1.0)] * 10
        try:
            from scipy.optimize import differential_evolution
            de_result = differential_evolution(
                de_objective_low_dim,
                bounds=bounds_low_dim,
                seed=100,
                maxiter=50,
                popsize=12,
                tol=1e-5,
                polish=False,
                init='sobol',
                workers=1
            )
            if de_result.fun < best_val:
                h_de = _low_dim_to_high_dim(de_result.x)
                h_de = _project_to_feasible(h_de)
                val_de = _correlation_value(h_de)
                if val_de < best_val:
                    best_val = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: Soft-max based L-BFGS-B refinement with decreasing temperature
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        for T in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0]:
            if time.time() > deadline - 10:
                break
            
            def soft_max_obj(h_raw):
                h = _project_to_feasible(h_raw)
                return _soft_max_objective(h, T)
            
            try:
                res = minimize(
                    soft_max_obj,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
                )
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Fine-grained coordinate descent polish
    if time.time() < deadline - 5:
        improved = True
        max_rounds = 50
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                current = best_h[i]
                trial_vals = np.clip(
                    current + np.array([-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1]),
                    0.0, 1.0
                )
                for trial_val in trial_vals:
                    h_trial = best_h.copy()
                    h_trial[i] = trial_val
                    h_trial = _project_to_feasible(h_trial)
                    val_trial = _correlation_value(h_trial)
                    if val_trial < best_val - 1e-12:
                        best_val = val_trial
                        best_h = h_trial.copy()
                        improved = True
    
    # Phase 4: Final high-precision polish with SLSQP
    if time.time() < deadline - 2:
        from scipy.optimize import minimize
        
        def slsqp_obj(h):
            return _correlation_value(_project_to_feasible(h))
        
        try:
            res = minimize(
                slsqp_obj,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.any(mask):
            best_h[mask] -= diff / np.sum(mask)
        else:
            best_h -= diff / N
        best_h = np.clip(best_h, 0, 1)
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
