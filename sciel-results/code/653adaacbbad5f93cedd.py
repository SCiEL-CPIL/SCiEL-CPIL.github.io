import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    
    def score(h):
        h = np.asarray(h)
        h = np.clip(h, 0, None)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        c = np.convolve(h_norm, h_norm, mode='full')
        c_max = np.max(c)
        if c_max < 1e-12:
            return 0.0
        return l2_sq / c_max

    def neg_score(h):
        return -score(h)

    best_h = np.zeros(n)
    best_val = 0.0
    
    # Generate diverse initial guesses
    starts = []
    
    # Random uniform
    for _ in range(20):
        starts.append(np.random.rand(n))
        
    # Sparse random
    for _ in range(20):
        h = np.zeros(n)
        k = np.random.randint(5, 25)
        idx = np.random.choice(n, k, replace=False)
        h[idx] = 1.0
        starts.append(h)
        
    # Structured: Alternating
    h_alt = np.zeros(n)
    h_alt[::2] = 1.0
    starts.append(h_alt)
    
    # Structured: Block
    h_block = np.zeros(n)
    h_block[:10] = 1.0
    starts.append(h_block)
    
    # Structured: Two peaks
    h_2peak = np.zeros(n)
    h_2peak[5] = 1.0
    h_2peak[40] = 1.0
    starts.append(h_2peak)

    for h0 in starts:
        # Normalize initial guess
        s = np.sum(h0)
        if s > 0:
            h0 = h0 / s
        else:
            continue
            
        try:
            res = minimize(neg_score, h0, method='Powell', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'maxfev': 3000})
            h_opt = res.x
            val = score(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            continue

    if best_val == 0.0:
        best_h = np.ones(n) / n
        best_val = score(best_h)
        
    # Normalize output to sum 1
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
        
    return list(best_h)
