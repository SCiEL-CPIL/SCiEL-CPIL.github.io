import numpy as np
from scipy.optimize import minimize

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [d, s, b]
    d: half-separation (centers at -d, d)
    s: width
    b: relative amplitude
    h = (1 * w1 + b * w2)^2
    w1, w2 are Gaussian pulses centered at -d, d with width s.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    d, s, b = params
    
    s = max(s, 0.001)
    d = min(max(d, 0.0), 0.25)
    b = max(0.0, b)
    
    c1 = -d
    c2 = d
    
    w1 = np.exp(-0.5 * ((x - c1) / s) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s) ** 2)
    
    a = 1.0
    f = a * w1 + b * w2
    
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
    conv = np.convolve(h, h, mode='full') * dx
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
        
    return l2_sq / (l1 * sup_norm)

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    best_score = -np.inf
    best_params = [0.25, 0.01, 1.0]
    
    # Grid search for d, s, b
    d_vals = np.linspace(0.1, 0.25, 10)
    s_vals = np.linspace(0.005, 0.05, 10)
    b_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    candidates = []
    
    for d in d_vals:
        for s in s_vals:
            for b in b_vals:
                params = [d, s, b]
                score = _compute_score(params)
                if score > best_score:
                    best_score = score
                    best_params = params
                candidates.append((score, params))
    
    # Refine top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for score, params in candidates[:10]:
        try:
            res = minimize(_neg_score, params, method='Nelder-Mead', 
                           bounds=([0.0, 0.001, 0.0], [0.25, 0.1, 1.0]),
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    d, s, b = best_params
    s = max(s, 0.001)
    d = min(max(d, 0.0), 0.25)
    b = max(0.0, min(b, 1.0))
    
    c1 = -d
    c2 = d
    
    w1 = np.exp(-0.5 * ((x - c1) / s) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s) ** 2)
    
    a = 1.0
    f = a * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
