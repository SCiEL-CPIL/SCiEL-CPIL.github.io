import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 135.0
_start_time = time.time()

def _score(h):
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _best_response(h, i, tol=1e-10):
    """Find optimal h_i given all other h_j fixed, via scalar optimization."""
    h = np.asarray(h, dtype=float).copy()
    
    def neg_score_i(x):
        h[i] = max(0.0, x)
        s = _score(h)
        return -s
    
    # Try multiple initial points for h_i
    best_val = -1e30
    best_x = 0.0
    
    # Get current sum and max_conv contributions
    h[i] = 0.0
    s1_rest = np.sum(h)
    conv_rest = np.convolve(h, h, mode='full')
    max_conv_rest = np.max(conv_rest) if len(conv_rest) > 0 else 0.0
    
    # h_i contributes to conv at positions i (from h_i * h_0) and 
    # various positions. The max of conv can be complex.
    # Just do a 1D search over h_i in [0, upper_bound]
    
    # Upper bound: if h_i is very large, s2 ~ h_i^2, s1 ~ h_i, max_conv ~ h_i * max(h)
    # So score ~ h_i^2 / (h_i * h_i * max_h) = 1/max_h, bounded.
    # But if h_i dominates, score approaches 1/max(h_rest) which could be small.
    # Actually let's just search numerically.
    
    # Initial guess: proportional to something
    if s1_rest > 1e-12:
        init_guess = s1_rest / N
    else:
        init_guess = 1.0 / N
    
    # Try a grid of initial points
    for x0 in [0.0, init_guess * 0.1, init_guess * 0.5, init_guess, init_guess * 2, init_guess * 5, init_guess * 10]:
        try:
            res = minimize(neg_score_i, np.array([x0]), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            x_opt = max(0.0, res.x[0])
            val = neg_score_i(x_opt)
            if val > best_val:
                best_val = val
                best_x = x_opt
        except Exception:
            pass
    
    return best_x

def construct():
    deadline = _start_time + _TIME_LIMIT - 5.0
    
    # Start with a reasonable initial configuration
    # Use a sparse pattern - alternate non-zero entries
    h = np.zeros(N)
    h[0] = 1.0
    h[10] = 0.5
    h[20] = 0.3
    h[30] = 0.2
    
    best_h = h.copy()
    best_s = _score(h)
    
    # Best-response dynamics
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            new_hi = _best_response(h, i)
            
            if abs(new_hi - h[i]) > 1e-12:
                h[i] = new_hi
                new_s = _score(h)
                if new_s > best_s:
                    best_s = new_s
                    best_h = h.copy()
                    no_improve_count = 0
                elif new_s > best_s - 1e-8:
                    no_improve_count = 0
                else:
                    # Revert if it made things worse
                    h[i] = h[i]
                    no_improve_count += 1
            
            if no_improve_count > N * 2:
                break
        
        if no_improve_count > N * 2:
            break
        
        iteration += 1
    
    # Try a few restarts with different initial patterns if time remains
    if time.time() < deadline - 10.0:
        restart_patterns = [
            lambda: (lambda h: (h.__setitem__(0, 1.0), h.__setitem__(25, 1.0)) and h)(np.zeros(N)),
            lambda: (lambda h: (h.__setitem__(5, 1.0), h.__setitem__(15, 1.0), h.__setitem__(25, 1.0)) and h)(np.zeros(N)),
            lambda: (lambda h: (h.__setitem__(0, 1.0), h.__setitem__(1, 1.0), h.__setitem__(2, 1.0)) and h)(np.zeros(N)),
        ]
        
        for init_fn in restart_patterns:
            if time.time() > deadline - 5.0:
                break
            
            h = init_fn()
            cur_s = _score(h)
            
            # Run best-response dynamics for this restart
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                new_hi = _best_response(h, i)
                h[i] = new_hi
                new_s = _score(h)
                if new_s > best_s:
                    best_s = new_s
                    best_h = h.copy()
    
    return [float(x) for x in best_h]
