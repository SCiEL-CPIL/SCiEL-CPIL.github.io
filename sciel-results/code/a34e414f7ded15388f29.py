import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _project_to_feasible(h):
    """Project h onto {sum(h)=N/2, 0<=h<=1} using iterative clipping and redistribution."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(500):
        diff = np.sum(h) - target
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
            h[mask] += (-diff) / np.sum(mask)
    return np.clip(h, 0.0, 1.0)

def _smooth_max_corr(h, temperature):
    """Smooth approximation of max(corr) using log-sum-exp."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    # Use log-sum-exp for numerical stability
    max_c = np.max(corr)
    if max_c < -1e300:
        return max_c
    shifted = corr - max_c
    log_sum_exp = max_c + np.log(np.sum(np.exp(shifted / temperature))) * temperature
    return log_sum_exp

def construct():
    start_time = time.time()
    h_seed = np.full(N, 0.5)
    best_h = h_seed.copy()
    best_value = _correlation_value(h_seed)
    
    # Temperature annealing schedule
    temps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    # Multiple random restarts for diversity
    rng = np.random.RandomState(42)
    
    for temp in temps:
        if time.time() - start_time > DEADLINE:
            break
        
        # Try a few restarts at this temperature
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Generate initial guess
            if r == 0:
                h_init = h_seed.copy()
            else:
                h_init = rng.uniform(0, 1, N)
                h_init = _project_to_feasible(h_init)
            
            # Objective with smooth max
            def objective(x):
                return _smooth_max_corr(x, temp)
            
            # Gradient of smooth max corr
            def gradient(x):
                h = x
                one_minus_h = 1.0 - h
                corr = np.correlate(h, one_minus_h, mode="full")
                max_c = np.max(corr)
                shifted = corr - max_c
                weights = np.exp(shifted / temp) / np.sum(np.exp(shifted / temp))
                
                # d(corr_k)/dh_i for each correlation term
                # corr = correlate(h, 1-h) = sum_j h_j * (1-h)_{j+k-N+1} for various shifts
                # Actually, let's compute gradient numerically or analytically
                # For efficiency, use finite difference approximation
                eps = 1e-7
                grad = np.zeros(N)
                for i in range(N):
                    h_plus = h.copy()
                    h_plus[i] += eps
                    h_plus = _project_to_feasible(h_plus)
                    f_plus = _smooth_max_corr(h_plus, temp)
                    
                    h_minus = h.copy()
                    h_minus[i] -= eps
                    h_minus = _project_to_feasible(h_minus)
                    f_minus = _smooth_max_corr(h_minus, temp)
                    
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                return grad
            
            # Use L-BFGS-B with bounds
            bounds = [(0.0, 1.0)] * N
            
            try:
                res = minimize(
                    objective,
                    h_init,
                    jac=gradient,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = res.x
                h_candidate = _project_to_feasible(h_candidate)
                
                # Check if valid
                if not np.all(np.isfinite(h_candidate)):
                    continue
                if abs(np.sum(h_candidate) - N/2.0) > 1e-6:
                    continue
                
                val = _correlation_value(h_candidate)
                if val < best_value:
                    best_value = val
                    best_h = h_candidate.copy()
                    
            except Exception:
                continue
    
    # Final polish with very low temperature
    if time.time() - start_time < DEADLINE - 10:
        try:
            def objective_final(x):
                return _smooth_max_corr(x, 1e-6)
            
            def gradient_final(x):
                h = x
                eps = 1e-7
                grad = np.zeros(N)
                for i in range(N):
                    h_plus = h.copy()
                    h_plus[i] += eps
                    h_plus = _project_to_feasible(h_plus)
                    f_plus = _smooth_max_corr(h_plus, 1e-6)
                    
                    h_minus = h.copy()
                    h_minus[i] -= eps
                    h_minus = _project_to_feasible(h_minus)
                    f_minus = _smooth_max_corr(h_minus, 1e-6)
                    
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                return grad
            
            bounds = [(0.0, 1.0)] * N
            res = minimize(
                objective_final,
                best_h,
                jac=gradient_final,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_candidate = _project_to_feasible(res.x)
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate) - N/2.0) <= 1e-6:
                val = _correlation_value(h_candidate)
                if val < best_value:
                    best_h = h_candidate
        except Exception:
            pass
    
    return best_h

# EVOLVE-BLOCK-END
