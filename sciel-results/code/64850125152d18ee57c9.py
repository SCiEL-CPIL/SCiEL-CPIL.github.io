import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 135.0  # safety margin before 150s kill

def _corr_vector(h):
    """Compute all lags of correlation."""
    return np.correlate(h, 1.0 - h, mode='full')

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = _corr_vector(h)
    return np.max(corr) / N * 2.0

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = np.sum(mask)
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    diff = TARGET_SUM - np.sum(h)
    if abs(diff) > 1e-15:
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _corr_gradient(h):
    """
    Compute gradient of correlation vector w.r.t. h.
    Returns array of shape (2N-1, N) where grad[k, j] = d(corr[k])/d(h[j]).
    
    np.correlate(h, 1-h, 'full')[k] = sum_i h[i] * (1-h[i+k-(N-1)])
    valid i: 0<=i<N, 0<=i+k-(N-1)<N
    """
    K = 2 * N - 1
    grad = np.zeros((K, N))
    for k in range(K):
        offset = k - (N - 1)
        # For each valid i, corr[k] includes h[i] * (1-h[i+offset])
        # d/dh[j]: 
        #   if j == i: (1-h[i+offset])
        #   if j == i+offset: -h[i]
        valid_i = np.arange(N)
        valid_j_offset = valid_i + offset
        valid_mask = (valid_j_offset >= 0) & (valid_j_offset < N)
        for i in valid_i[valid_mask]:
            j_off = i + offset
            grad[k, i] += (1.0 - h[j_off])
            grad[k, j_off] += (-h[i])
    return grad

def _epigraph_objective(x):
    """
    Epigraph formulation: minimize t
    subject to corr[k] <= t for all k,
               0 <= h <= 1,
               sum(h) = N/2
    
    x = [h_0, ..., h_{N-1}, t]
    """
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    return t / N * 2.0

def _epigraph_grad(x):
    """Gradient of epigraph objective."""
    h = x[:N]
    t = x[N]
    # Objective is just t/N*2, so gradient w.r.t. t is 1/N*2
    # But we need to account for constraints via the active set
    # For SLSQP, we provide the gradient of the objective function
    grad = np.zeros(N + 1)
    grad[N] = 2.0 / N
    return grad

def _epigraph_constraints(x):
    """
    Constraints for epigraph formulation.
    x = [h_0, ..., h_{N-1}, t]
    """
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    K = len(corr)
    
    # Inequality constraints: corr[k] - t <= 0 for all k
    # We can't pass 199 constraints to SLSQP efficiently, so use a smooth max
    # Or we can use the fact that SLSQP handles multiple constraints
    
    # Actually, let's just use the smooth surrogate approach but with analytical gradient
    pass

def _smooth_max_corr(h, beta=0.05):
    """Smooth approximation of max(corr) using log-sum-exp."""
    corr = _corr_vector(h)
    c_max = np.max(corr)
    shifted = corr - c_max
    return c_max + beta * np.log(np.sum(np.exp(shifted / beta)))

def _smooth_max_corr_grad(h, beta=0.05):
    """
    Analytical gradient of smooth max of correlation w.r.t. h.
    Uses log-sum-exp trick with analytical gradient of corr.
    """
    corr = _corr_vector(h)
    K = len(corr)
    c_max = np.max(corr)
    shifted = corr - c_max
    exp_shifted = np.exp(shifted / beta)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of log-sum-exp: sum_k weights[k] * d(corr[k])/dh
    # We need d(corr[k])/dh for all k, weighted by weights[k]
    # corr[k] = sum_i h[i] * (1-h[i+k-(N-1)])
    # d(corr[k])/dh[j] = sum over valid i: 
    #   if j==i: (1-h[i+offset_k])
    #   if j==i+offset_k: -h[i]
    
    grad = np.zeros(N)
    for k in range(K):
        if weights[k] < 1e-15:
            continue
        offset = k - (N - 1)
        for i in range(N):
            j_off = i + offset
            if 0 <= j_off < N:
                grad[i] += weights[k] * (1.0 - h[j_off])
                grad[j_off] += weights[k] * (-h[i])
    
    return grad

def _smooth_objective(h, beta=0.05):
    """Smooth approximation of the max correlation value."""
    return _smooth_max_corr(h, beta) / N * 2.0

def _smooth_objective_grad(h, beta=0.05):
    """Gradient of the smooth objective."""
    return _smooth_max_corr_grad(h, beta) / N * 2.0

def _coordinate_polish(h, max_iters=100, tol=1e-12):
    """
    Coordinate descent: for each i, optimize h[i] while maintaining sum constraint.
    """
    h = _project_feasible(h).copy()
    current_val = _correlation_max(h)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Choose j to compensate (pick j with h[j] closest to 0.5 for flexibility)
            j = np.argmin(np.abs(h - 0.5))
            if j == i:
                j = (i + 1) % N
            
            def obj_1d(x_i):
                h_j = TARGET_SUM - (np.sum(h) - h[i]) + x_i
                if h_j < 0.0 or h_j > 1.0:
                    return 1e10
                h_temp = h.copy()
                h_temp[i] = x_i
                h_temp[j] = h_j
                return _correlation_max(h_temp)
            
            sum_others = np.sum(h) - h[i]
            x_min = max(0.0, -(TARGET_SUM - sum_others))
            x_max = min(1.0, 1.0 - (TARGET_SUM - sum_others))
            x_min = max(0.0, x_min)
            x_max = min(1.0, x_max)
            
            if x_min >= x_max:
                continue
            
            try:
                res = minimize_scalar(obj_1d, bounds=(x_min, x_max), method='bounded',
                                     options={'xatol': tol})
                if res.success:
                    h_j = TARGET_SUM - (np.sum(h) - h[i]) + res.x
                    if 0.0 <= h_j <= 1.0:
                        h_temp = h.copy()
                        h_temp[i] = res.x
                        h_temp[j] = h_j
                        val_temp = _correlation_max(h_temp)
                        if val_temp < current_val - tol:
                            h = h_temp
                            current_val = val_temp
                            improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return _project_feasible(h)

def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Phase 1: Differential Evolution for global search
    if time.time() - start_time < DEADLINE - 80:
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def de_objective(x):
            h = _project_feasible(x)
            return _smooth_objective(h, beta=0.1)
        
        try:
            res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                init='latinhypercube',
                update_interval=10,
                polish=True,
                workers=-1
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Phase 2: SLSQP with analytical gradient from multiple starting points
    if time.time() - start_time < DEADLINE - 40:
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
        
        # Try from current best with analytical gradient
        for beta in [0.05, 0.03, 0.02]:
            if time.time() - start_time > DEADLINE - 30:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, beta),
                    best_h,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    jac=lambda x: _smooth_objective_grad(x, beta),
                    options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        # Try from alternating patterns
        for base in [0.1, 0.2, 0.3, 0.4]:
            if time.time() - start_time > DEADLINE - 30:
                break
            alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
            h_alt = _project_feasible(alt)
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 0.05),
                    h_alt,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    jac=lambda x: _smooth_objective_grad(x, 0.05),
                    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        # Try from sinusoidal patterns
        for freq in [1, 2, 3, 5]:
            if time.time() - start_time > DEADLINE - 25:
                break
            t_arr = np.arange(N)
            for amp in [0.2, 0.3, 0.4]:
                for phase in [0, np.pi/2, np.pi]:
                    h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t_arr / N + phase)
                    h_sin = _project_feasible(h_sin)
                    try:
                        res = minimize(
                            lambda x: _smooth_objective(x, 0.05),
                            h_sin,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=[constraint],
                            jac=lambda x: _smooth_objective_grad(x, 0.05),
                            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                        )
                        h_res = _project_feasible(res.x)
                        val = _correlation_max(h_res)
                        if val < best_val - 1e-10:
                            best_val = val
                            best_h = h_res.copy()
                    except Exception:
                        pass
    
    # Phase 3: Multi-start local refinement with analytical gradients
    if time.time() - start_time < DEADLINE - 20:
        # Generate diverse starting points
        rng = np.random.RandomState(123)
        num_starts = 20
        
        for s in range(num_starts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Random perturbation of best solution
            h_start = best_h.copy()
            h_start += rng.normal(0, 0.1, N)
            h_start = _project_feasible(h_start)
            
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 0.04),
                    h_start,
                    method='SLSQP',
                    bounds=[(0.0, 1.0) for _ in range(N)],
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 
                                  'jac': lambda x: np.ones(N)}],
                    jac=lambda x: _smooth_objective_grad(x, 0.04),
                    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Phase 4: Coordinate descent polish
    if time.time() - start_time < DEADLINE - 10:
        h_cd = _coordinate_polish(best_h, max_iters=50, tol=1e-12)
        val_cd = _correlation_max(h_cd)
        if val_cd < best_val - 1e-12:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Phase 5: Final high-precision SLSQP with tighter beta
    if time.time() - start_time < DEADLINE - 5:
        for beta in [0.02, 0.01]:
            if time.time() - start_time > DEADLINE - 3:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, beta),
                    best_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0) for _ in range(N)],
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM,
                                  'jac': lambda x: np.ones(N)}],
                    jac=lambda x: _smooth_objective_grad(x, beta),
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Final coordinate polish
    if time.time() - start_time < DEADLINE - 2:
        h_final = _coordinate_polish(best_h, max_iters=30, tol=1e-14)
        val_final = _correlation_max(h_final)
        if val_final < best_val - 1e-14:
            best_val = val_final
            best_h = h_final.copy()
    
    # Ensure final validity
    best_h = _project_feasible(best_h)
    
    return best_h
