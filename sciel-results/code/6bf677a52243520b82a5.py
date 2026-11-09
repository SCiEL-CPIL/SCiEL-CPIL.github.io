import numpy as np
from scipy.optimize import minimize
import time

def compute_probability(support, weights):
    n = len(support)
    s1 = support[:, np.newaxis, np.newaxis, np.newaxis]
    s2 = support[np.newaxis, :, np.newaxis, np.newaxis]
    s3 = support[np.newaxis, np.newaxis, :, np.newaxis]
    s4 = support[np.newaxis, np.newaxis, np.newaxis, :]
    
    w1 = weights[:, np.newaxis, np.newaxis, np.newaxis]
    w2 = weights[np.newaxis, :, np.newaxis, np.newaxis]
    w3 = weights[np.newaxis, np.newaxis, :, np.newaxis]
    w4 = weights[np.newaxis, np.newaxis, np.newaxis, :]
    
    mask = (s1 + s2 + s3) < (2.0 * s4)
    p = np.sum(w1 * w2 * w3 * w4 * mask)
    return p

def objective(params, n):
    v = params[:n]
    u = params[n:]
    
    # Clip to prevent overflow
    v = np.clip(v, -20, 20)
    u = np.clip(u, -20, 20)
    
    s = np.zeros(n)
    s[0] = np.exp(v[0])
    for i in range(1, n):
        s[i] = s[i-1] + np.exp(v[i])
        
    u_max = np.max(u)
    w_unnorm = np.exp(u - u_max)
    w_sum = np.sum(w_unnorm)
    if w_sum <= 0 or not np.isfinite(w_sum):
        return 1.0 # Penalty (minimize -p, so 1.0 is bad since p <= 1)
        
    w = w_unnorm / w_sum
    
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        return 1.0
        
    p = compute_probability(s, w)
    
    if not np.isfinite(p):
        return 1.0
        
    return -p

def optimize_for_n(n, initial_params=None, max_iter=2000):
    if initial_params is None:
        v = np.zeros(n)
        # Target support approx: 0, 1, 3, 9
        # s0 = 0.1, s1 = 1, s2 = 3, s3 = 9
        # gaps: 0.9, 2, 6
        # v0 = log(0.1)
        # v1 = log(0.9)
        # v2 = log(2)
        # v3 = log(6)
        target_s = [0.1, 1.0, 3.0, 9.0, 27.0, 81.0, 243.0, 729.0]
        for i in range(n):
            if i == 0:
                v[i] = np.log(target_s[0])
            else:
                gap = target_s[i] - target_s[i-1]
                if gap <= 0:
                    gap = 1.0
                v[i] = np.log(gap)
                
        u = np.zeros(n)
        for i in range(n):
            u[i] = -0.5 * i
            
        initial_params = np.concatenate([v, u])
        
    res = minimize(objective, initial_params, args=(n,), method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'disp': False, 'xtol': 1e-10, 'ftol': 1e-10})
    
    best_params = res.x
    v = np.clip(best_params[:n], -20, 20)
    u = np.clip(best_params[n:], -20, 20)
    
    s = np.zeros(n)
    s[0] = np.exp(v[0])
    for i in range(1, n):
        s[i] = s[i-1] + np.exp(v[i])
        
    u_max = np.max(u)
    w_unnorm = np.exp(u - u_max)
    w_sum = np.sum(w_unnorm)
    if w_sum <= 0:
        return None, None, -1.0
        
    w = w_unnorm / w_sum
    
    p = compute_probability(s, w)
    
    return s, w, p

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_p = -1.0
    best_s = None
    best_w = None
    
    # Try n=3, 4, 5
    for n in [3, 4, 5]:
        if time.time() > deadline - 30:
            break
            
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 30:
                break
                
            try:
                # Perturb initial guess slightly for restarts > 0
                init = None
                if restart > 0:
                    # Use previous best if available? Or just random perturbation of base
                    pass # For simplicity, use same base, Nelder-Mead is stochastic enough with different starting points if we jitter
                    # Let's just add small noise to the standard init
                    v_base = np.zeros(n)
                    target_s = [0.1, 1.0, 3.0, 9.0, 27.0, 81.0, 243.0, 729.0]
                    for i in range(n):
                        if i == 0:
                            v_base[i] = np.log(target_s[0])
                        else:
                            gap = target_s[i] - target_s[i-1]
                            if gap <= 0: gap = 1.0
                            v_base[i] = np.log(gap)
                    u_base = np.zeros(n)
                    for i in range(n):
                        u_base[i] = -0.5 * i
                    base_params = np.concatenate([v_base, u_base])
                    noise = np.random.normal(0, 0.1, size=2*n)
                    init = base_params + noise
                else:
                    init = None # Use default in function
                
                s, w, p = optimize_for_n(n, initial_params=init)
                if s is not None and p > best_p:
                    best_p = p
                    best_s = s
                    best_w = w
            except Exception:
                pass
                
    # Fallback
    if best_s is None:
        best_s = np.array([0.0, 1.0, 3.0])
        best_w = np.array([0.5, 0.3, 0.2])
        best_w = best_w / np.sum(best_w)
        
    # Validate
    if not (np.all(best_s >= 0) and np.all(best_w >= 0) and np.sum(best_w) > 1e-9):
        n = 32
        best_s = np.arange(n, dtype=np.float64)
        best_w = np.ones(n, dtype=np.float64) / n

    return best_s, best_w
