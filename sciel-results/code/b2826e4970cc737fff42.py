import numpy as np
from scipy.optimize import minimize

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score_symmetric(c, w):
    """
    Computes score for two symmetric Gaussian pulses at -c and c with width w.
    h = exp(-0.5*((x-c)/w)^2) + exp(-0.5*((x+c)/w)^2)
    Note: The current program uses h = (a*w1 + b*w2)^2.
    If a=b=1, h = (w1 + w2)^2 = w1^2 + w2^2 + 2 w1 w2.
    If disjoint, w1 w2 ~ 0, so h ~ w1^2 + w2^2.
    This is equivalent to two Gaussian squares (which are also Gaussian-like, width w/sqrt(2)).
    Let's stick to the form in the current program to be safe, but optimize only c and w.
    """
    # h = (w1 + w2)^2
    w1 = np.exp(-0.5 * ((x - c) / w) ** 2)
    w2 = np.exp(-0.5 * ((x + c) / w) ** 2)
    f = w1 + w2
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

def _neg_score_symmetric(params):
    c, w = params
    if c < 0.01 or w < 0.001:
        return 1e10
    # Ensure they are somewhat disjoint? 
    # If c is small, they overlap. 
    # The score function will naturally penalize bad overlaps if that lowers the ratio.
    s = _compute_score_symmetric(c, w)
    if s < 0:
        return 1e10
    return -s

def construct():
    # Optimize c and w
    # Initial guess
    x0 = [0.2, 0.05]
    bounds = [(0.05, 0.25), (0.005, 0.2)]
    
    best_score = -np.inf
    best_params = x0
    
    # Try multiple starting points to avoid local minima
    start_points = [
        [0.1, 0.05],
        [0.2, 0.05],
        [0.25, 0.1],
        [0.15, 0.02],
        [0.2, 0.1],
        [0.1, 0.1]
    ]
    
    for sp in start_points:
        try:
            res = minimize(_neg_score_symmetric, sp, bounds=bounds, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < -best_score: # -res.fun > best_score
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # If optimization failed, fallback to a grid search
    if best_score < 0:
        for c in np.linspace(0.05, 0.25, 50):
            for w in np.linspace(0.005, 0.2, 50):
                s = _compute_score_symmetric(c, w)
                if s > best_score:
                    best_score = s
                    best_params = [c, w]

    c, w = best_params
    c = np.clip(c, 0.0, 0.25)
    w = max(w, 0.001)
    
    w1 = np.exp(-0.5 * ((x - c) / w) ** 2)
    w2 = np.exp(-0.5 * ((x + c) / w) ** 2)
    
    f = w1 + w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
