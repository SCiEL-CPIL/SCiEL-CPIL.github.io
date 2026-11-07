import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _raw_max_corr(h):
    """Compute just the max correlation value (for optimization)."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(100):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _smooth_max_corr(h, lam=50.0):
    """Smooth approximation of max(corr) using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) * lam))) / lam

def _local_search(h, max_iters=2000, time_limit=None, step_scale=1.0):
    """
    Local search with pair moves.
    """
    best_h = h.copy()
    best_val = _raw_max_corr(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 15.0
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Try several random pair moves
        improved = False
        for _ in range(20):
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Random direction and magnitude
            d = np.random.uniform(-0.1 * step_scale, 0.1 * step_scale)
            
            new_h_i = best_h[i] + d
            new_h_j = best_h[j] - d
            
            if 0 <= new_h_i <= 1 and 0 <= new_h_j <= 1:
                new_h = best_h.copy()
                new_h[i] = new_h_i
                new_h[j] = new_h_j
                val = _raw_max_corr(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
                    improved = True
                    break
        
        if not improved and it > 100:
            # Try smaller steps
            step_scale *= 0.7
            if step_scale < 0.01:
                break
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution on Fourier Coefficients
    # Use more harmonics for richer shapes
    n_harm = 15
    n_params = 2 * n_harm
    i = np.arange(N)
    
    cols = []
    for k in range(1, n_harm + 1):
        cols.append(np.cos(2 * np.pi * k * i / N))
        cols.append(np.sin(2 * np.pi * k * i / N))
    basis = np.column_stack(cols)
    
    def fourier_objective(params):
        h = np.full(N, 0.5)
        h += params @ basis
        h = _project_to_feasible(h)
        return _raw_max_corr(h)
    
    bounds = [(-0.5, 0.5)] * n_params
    
    de_deadline = start_time + 40.0
    
    def timed_fourier_obj(params):
        if time.time() > de_deadline:
            raise TimeoutError()
        return fourier_objective(params)
    
    try:
        result = differential_evolution(
            timed_fourier_obj,
            bounds=bounds,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None:
            h_de = 0.5 + result.x @ basis
            h_de = _project_to_feasible(h_de)
            val_de = _raw_max_corr(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Local Search Polish
    ls_deadline = deadline - 20.0
    if time.time() < ls_deadline:
        best_h = _local_search(best_h, time_limit=ls_deadline)
        best_val = _raw_max_corr(best_h)
    
    # Phase 3: SLSQP with smooth objective - try multiple lambda values
    if time.time() < deadline - 8.0:
        try:
            for lam in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 4.0:
                    break
                
                def smooth_obj(x, lam=lam):
                    return _smooth_max_corr(x, lam=lam)
                
                def constraint(x):
                    return np.sum(x) - N / 2.0
                
                res = minimize(
                    smooth_obj,
                    best_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': constraint}],
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res.success or res.fun < best_val:
                    h_opt = _project_to_feasible(res.x)
                    val_opt = _raw_max_corr(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision local search
    if time.time() < deadline - 1.0:
        best_h = _local_search(best_h, max_iters=3000, time_limit=deadline - 0.5, step_scale=0.5)
        best_val = _raw_max_corr(best_h)
        
    best_h = _project_to_feasible(best_h)
    
    return best_h
