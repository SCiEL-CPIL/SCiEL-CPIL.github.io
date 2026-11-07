import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2"""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h to [0,1] and sum=N/2 using bisection on a global shift"""
    target = N / 2.0
    
    lo = -np.max(h) - 1.0
    hi = 1.0 - np.min(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h + mid, 0.0, 1.0)) - target
        if val > 0:
            hi = mid
        else:
            lo = mid
            
    s = (lo + hi) / 2.0
    h_proj = np.clip(h + s, 0.0, 1.0)
    
    diff = np.sum(h_proj) - target
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmax(h_proj)
        else:
            idx = np.argmin(h_proj)
        h_proj[idx] -= diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj

def _soft_max(corr, lam):
    """Soft max using log-sum-exp for numerical stability"""
    m = np.max(corr)
    if m == 0:
        return 0.0
    shifted = lam * (corr - m)
    s = np.sum(np.exp(shifted))
    if s == 0:
        return m
    return m + np.log(s) / lam

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial best: Seed
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Define the objective function factory
    def make_obj(lam):
        def obj(x):
            if time.time() > deadline - 5:
                return 1e9
            h = _project_to_feasible(x)
            corr = np.correlate(h, 1 - h, mode="full")
            return _soft_max(corr, lam)
        return obj
    
    bounds = [(0.0, 1.0)] * N
    
    # Initializations
    inits = [
        np.full(N, 0.5),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        np.sin(np.linspace(0, 2*np.pi, N)) * 0.5 + 0.5,
        np.cos(np.linspace(0, 2*np.pi, N)) * 0.5 + 0.5,
        np.sign(np.sin(np.linspace(0, 2*np.pi, N))) * 0.5 + 0.5,
        np.random.RandomState(42).uniform(0, 1, N),
        np.random.RandomState(123).uniform(0, 1, N),
    ]
    
    inits = [_project_to_feasible(x) for x in inits]
    
    current_h = best_h.copy()
    
    # Ramp up lambda
    for lam in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 15:
            break
        
        obj = make_obj(lam)
        
        try:
            res = minimize(
                obj,
                current_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                current_h = best_h.copy()
            
        except Exception:
            pass
        
        if time.time() < deadline - 30:
            init_idx = int(time.time() * 1000) % len(inits)
            x0 = inits[init_idx].copy()
            x0 += np.random.RandomState(init_idx).normal(0, 0.1, N)
            x0 = _project_to_feasible(x0)
            
            try:
                res = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
                )
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    current_h = best_h.copy()
                    
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 5:
        lam_final = 10000.0
        obj_final = make_obj(lam_final)
        try:
            res = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    if time.time() < deadline - 2:
        lam_final2 = 20000.0
        obj_final2 = make_obj(lam_final2)
        try:
            res = minimize(
                obj_final2,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    best_h = _project_to_feasible(best_h)
    
    return best_h
