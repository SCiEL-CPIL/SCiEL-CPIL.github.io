import numpy as np
import time
from scipy.optimize import minimize

def _union_length_1d(a, c, N):
    """Compute the 1D union length of intervals [a[i] + c*i/N, a[i] + c*i/N + 1/N] for i in range(N)."""
    starts = a + c * np.arange(N) / N
    ends = starts + 1.0 / N
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = ends[order]
    
    length = 0.0
    curr_start = s_sorted[0]
    curr_end = e_sorted[0]
    for k in range(1, N):
        if s_sorted[k] <= curr_end:
            if e_sorted[k] > curr_end:
                curr_end = e_sorted[k]
        else:
            length += curr_end - curr_start
            curr_start = s_sorted[k]
            curr_end = e_sorted[k]
    length += curr_end - curr_start
    return length

def _compute_volume_sep(a, num_c=100):
    """
    Compute the volume of the union of tubes assuming separable structure:
    x-positions depend only on i (via a[i]), y-positions depend only on j (via a[j]).
    This exploits the fact that at height c, the x-section is the union of intervals
    determined by a[i] + c*i/N, and similarly for y. The area is product of lengths.
    """
    N = 8
    c_vals = np.linspace(0, 1, num_c)
    total = 0.0
    for c in c_vals:
        len_x = _union_length_1d(a, c, N)
        # Since y-positions are the same array a, len_y is identical
        len_y = len_x
        total += len_x * len_y
    return total / num_c

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    N = 8
    
    # Objective function for L-BFGS-B
    def objective(x, num_c=100):
        if time.time() > deadline - 5:
            return 1e9
        a = np.zeros(N)
        a[0] = 0.0
        a[1:] = x
        return _compute_volume_sep(a, num_c=num_c)
        
    x0 = np.zeros(N-1)
    
    best_x = x0.copy()
    best_val = objective(x0, 200)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated Annealing for global exploration
    # We search in the space of a[1..7] (a[0] is fixed at 0)
    # The search space is 7-dimensional.
    T = 0.05
    cooling = 0.999
    min_T = 1e-6
    max_iter_sa = 50000
    
    current_x = x0.copy()
    current_val = best_val
    current_a = np.concatenate([[0.0], current_x])
    
    for it in range(max_iter_sa):
        if time.time() > deadline - 120:
            break
        if T < min_T:
            break
            
        # Perturb
        step_size = T * 0.5
        delta = rng.normal(0, step_size, size=N-1)
        trial_x = current_x + delta
        trial_a = np.concatenate([[0.0], trial_x])
        
        # Evaluate
        trial_val = _compute_volume_sep(trial_a, num_c=50) # Lower precision for speed
        
        delta_val = trial_val - current_val
        
        # Acceptance probability
        if delta_val < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_val / T)
            
        if accept:
            current_x = trial_x
            current_val = trial_val
            if current_val < best_val:
                best_val = current_val
                best_x = trial_x.copy()
                
        T *= cooling
        
    # Phase 2: Local Refinement with L-BFGS-B
    # We use multiple restarts from the best SA result and random perturbations
    
    def cost_func_lbgfs(x, num_c=200):
        if time.time() > deadline - 5:
            return 1e9
        a = np.zeros(N)
        a[0] = 0.0
        a[1:] = x
        return _compute_volume_sep(a, num_c=num_c)
        
    # Run several L-BFGS-B optimizations
    for restart in range(10):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            x_curr = best_x
        else:
            # Add small noise to best_x for restart
            x_curr = best_x + rng.normal(0, 0.01, size=N-1)
            
        try:
            res = minimize(cost_func_lbgfs, x_curr, args=(200,), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Phase 3: Final high-precision polish pass
    # Use a higher number of c-samples and a more robust optimizer if needed
    if time.time() < deadline - 30:
        def cost_func_high_res(x):
            if time.time() > deadline - 5:
                return 1e9
            a = np.zeros(N)
            a[0] = 0.0
            a[1:] = x
            return _compute_volume_sep(a, num_c=500)
            
        try:
            # Nelder-Mead can be more robust to non-smoothness at the very end
            res = minimize(cost_func_high_res, best_x, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # One more L-BFGS-B pass with high resolution
        if time.time() < deadline - 10:
            try:
                res = minimize(cost_func_high_res, best_x, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    a_opt = np.zeros(N)
    a_opt[0] = 0.0
    a_opt[1:] = best_x
    
    i_k = np.arange(64) // N
    j_k = np.arange(64) % N
    
    x_pos = a_opt[i_k]
    y_pos = a_opt[j_k]
    
    positions = np.stack([x_pos, y_pos], axis=1)
    
    return positions
