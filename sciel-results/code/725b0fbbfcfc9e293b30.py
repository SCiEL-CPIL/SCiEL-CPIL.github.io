import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    """
    Computes the objective value for a given height vector h.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0
    
    # Scale invariance: value is invariant under h -> c*h.
    # We normalize h to sum to 1 for numerical stability in optimization, 
    # but the value calculation itself is scale invariant.
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _tent_to_h(tent_params):
    """
    Convert tent basis parameters to a height vector of length N.
    
    tent_params structure:
    - num_tents = len(tent_params) // 4
    - For each tent i:
        - center (float, 0 to N-1)
        - width (float, > 0)
        - amplitude (float, > 0)
        - (padding for fixed size if needed, but we use variable length in list)
    
    Actually, for scipy.optimize, fixed length is better.
    Let's fix the number of tents to K=4.
    Parameters: [c0, w0, a0, c1, w1, a1, c2, w2, a2, c3, w3, a3]
    Total 12 parameters.
    
    Non-negativity is enforced by using positive amplitudes and the max(0, ...) in tent formula.
    We will optimize log(amplitude) and log(width) to ensure positivity? 
    Or just use bounds in the optimizer?
    Nelder-Mead doesn't support bounds.
    So we transform variables:
    a_i = exp(a_i_log)
    w_i = exp(w_i_log)
    c_i is free (center)
    
    So 12 parameters:
    4 centers (free)
    4 log-widths (free)
    4 log-amplitudes (free)
    """
    K = 4
    centers = tent_params[0:K]
    log_widths = tent_params[K:2*K]
    log_amps = tent_params[2*K:3*K]
    
    widths = np.exp(log_widths)
    amps = np.exp(log_amps)
    
    x = np.arange(N)
    h = np.zeros(N)
    
    for i in range(K):
        c = centers[i]
        w = widths[i]
        a = amps[i]
        
        # Tent function: a * max(0, 1 - |x - c| / w)
        # To avoid division by zero or tiny widths causing spikes, 
        # the exp ensures w > 0.
        h += a * np.maximum(0, 1 - np.abs(x - c) / w)
        
    return h

def _objective_tent(tent_params):
    """
    Objective function for Nelder-Mead minimization (returns negative value).
    """
    try:
        h = _tent_to_h(tent_params)
        # h is non-negative by construction (sum of positive tents)
        val = _compute_value(h)
        return -val
    except Exception:
        return 1e10 # Large penalty for failure

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Strategy:
    # 1. Generate a diverse set of initial tent configurations.
    # 2. Evaluate them.
    # 3. Polish the top candidates using Nelder-Mead on the 12D tent parameter space.
    
    K = 4 # Number of tents
    num_params = 3 * K
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform-like: 4 tents evenly spaced, wide, equal height
    # c: 6, 19, 31, 44 (approx N/8, 3N/8, 5N/8, 7N/8)
    # w: 10, 10, 10, 10
    # a: 1, 1, 1, 1
    c0 = np.array([6, 19, 31, 44])
    w0 = np.array([10, 10, 10, 10])
    a0 = np.array([1, 1, 1, 1])
    p0 = np.concatenate([c0, np.log(w0), np.log(a0)])
    candidates.append(p0)
    
    # 2. Single wide tent (Gaussian-like approx)
    c1 = np.array([25, 25, 25, 25])
    w1 = np.array([20, 20, 20, 20])
    a1 = np.array([1, 1, 1, 1])
    p1 = np.concatenate([c1, np.log(w1), np.log(a1)])
    candidates.append(p1)
    
    # 3. Two sharp peaks
    c2 = np.array([10, 40, 10, 40])
    w2 = np.array([5, 5, 5, 5])
    a2 = np.array([1, 1, 1, 1])
    p2 = np.concatenate([c2, np.log(w2), np.log(a2)])
    candidates.append(p2)
    
    # 4. Random starts
    np.random.seed(50)
    for _ in range(20):
        c = np.random.uniform(0, N, K)
        w = np.random.uniform(2, 20, K)
        a = np.random.uniform(0.1, 5.0, K)
        p = np.concatenate([c, np.log(w), np.log(a)])
        candidates.append(p)
        
    # Evaluate all candidates
    best_val = -np.inf
    best_params = None
    
    for p in candidates:
        val = _compute_value(_tent_to_h(p))
        if val > best_val:
            best_val = val
            best_params = p.copy()
            
    # Polish the best candidate using Nelder-Mead
    # Nelder-Mead is robust for low dimensions (12D is fine)
    
    # We will also try polishing the top 3 distinct candidates to avoid local minima
    
    # Sort candidates by value
    scored = []
    for p in candidates:
        val = _compute_value(_tent_to_h(p))
        scored.append((val, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Top 5 candidates to polish
    top_candidates = [p for _, p in scored[:5]]
    
    for p_init in top_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective_tent,
                p_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < -best_val:
                val_opt = -res.fun
                if val_opt > best_val:
                    best_val = val_opt
                    best_params = res.x.copy()
        except Exception:
            pass
            
    # Final check
    if best_params is None:
        # Fallback to uniform
        return list(np.ones(N) / N)
        
    h_best = _tent_to_h(best_params)
    
    # Ensure non-negative and finite
    h_best = np.clip(h_best, 0, None)
    if np.sum(h_best) == 0:
        h_best = np.ones(N) / N
        
    return list(h_best)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
