import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0

def _max_overlap(h):
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    return 0.5 - np.min(autocorr)

def _soft_max_overlap(h, temperature=0.05):
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    overlap = 0.5 - autocorr
    max_o = np.max(overlap)
    log_sum = max_o + np.log(np.sum(np.exp((overlap - max_o) / temperature))) * temperature
    return log_sum

def construct():
    start = time.time()
    
    # Initialize with 0.5
    h = np.full(N, 0.5)
    
    # Try to optimize using L-BFGS-B on a parameterization that respects bounds
    # h = sigmoid(x) * 2, then scale to sum N/2?
    # Or just use bounds in L-BFGS-B directly on h.
    
    def obj(x):
        h_x = np.clip(x, 0, 1)
        s = np.sum(h_x)
        if s > 1e-10:
            h_x = h_x * (N/2.0) / s
            h_x = np.clip(h_x, 0, 1)
        return _soft_max_overlap(h_x, 0.05)
    
    # Start from 0.5
    x0 = np.full(N, 0.5)
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', 
                          bounds=[(0, 1)]*N,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        h_opt = np.clip(result.x, 0, 1)
        s_opt = np.sum(h_opt)
        if s_opt > 0:
            h_opt *= (N/2.0) / s_opt
            h_opt = np.clip(h_opt, 0, 1)
        
        if time.time() - start < DEADLINE - 10:
            # Try multiple random restarts
            best_h = h_opt
            best_score = _max_overlap(h_opt)
            
            for i in range(10):
                if time.time() - start > DEADLINE - 10:
                    break
                np.random.seed(i)
                x0_r = np.random.uniform(0, 1, N)
                x0_r *= (N/2.0) / np.sum(x0_r)
                x0_r = np.clip(x0_r, 0, 1)
                
                res_r = minimize(obj, x0_r, method='L-BFGS-B', 
                                 bounds=[(0, 1)]*N,
                                 options={'maxiter': 200, 'ftol': 1e-10})
                h_r = np.clip(res_r.x, 0, 1)
                s_r = np.sum(h_r)
                if s_r > 0:
                    h_r *= (N/2.0) / s_r
                    h_r = np.clip(h_r, 0, 1)
                score_r = _max_overlap(h_r)
                if score_r < best_score:
                    best_h = h_r
                    best_score = score_r
            
            h_final = best_h
        else:
            h_final = h_opt
    except Exception:
        h_final = np.full(N, 0.5)
        
    # Ensure constraints
    h_final = np.clip(h_final, 0, 1)
    s = np.sum(h_final)
    if s > 0:
        h_final *= (N/2.0) / s
        h_final = np.clip(h_final, 0, 1)
        
    # Fix sum
    diff = N/2.0 - np.sum(h_final)
    if abs(diff) > 1e-10:
        idx = np.argmin(h_final) if diff > 0 else np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0, 1)
        
    return h_final
