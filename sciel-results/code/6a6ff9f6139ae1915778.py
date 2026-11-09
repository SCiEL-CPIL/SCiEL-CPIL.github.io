import numpy as np
from scipy.optimize import minimize
import time

def _score_from_heights(h):
    """
    Computes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for discrete heights.
    Assumes unit interval width (dt=1) for shape optimization, 
    as the ratio is scale-invariant with respect to the number of points 
    if we consider the continuous limit properly, or rather, the discrete 
    sum approximations cancel out.
    
    Specifically:
    L2_sq = sum(h^2)
    L1 = sum(h)
    Autoconv = np.convolve(h, h, 'full')
    Peak = max(Autoconv)
    
    Ratio = L2_sq / (L1 * Peak)
    """
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    
    l2_sq = np.sum(h ** 2)
    
    # Compute autoconvolution
    # For non-negative h, the peak of the autoconvolution is at the center (index N-1)
    # if h is symmetric, but generally it's the max.
    ac = np.convolve(h, h, mode='full')
    peak = np.max(ac)
    
    if peak <= 0:
        return 0.0
        
    return l2_sq / (l1 * peak)

def _neg_ratio(h):
    """Objective function for minimization: negative of the score."""
    return -_score_from_heights(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    N = 50
    best_score = -np.inf
    best_h = None
    
    # We will optimize the heights directly.
    # To handle the non-negativity constraint and scale invariance,
    # we can parameterize h_i = exp(x_i) or h_i = x_i^2.
    # However, direct optimization with bounds is often more robust.
    # Let's use L-BFGS-B or Powell with bounds [0, 1].
    # Since the function is scale invariant, we should fix the scale to avoid
    # the optimizer drifting to 0 or infinity.
    # We can fix sum(h) = 1.
    
    # Strategy:
    # 1. Multiple random restarts.
    # 2. Use parameterization h_i = exp(y_i) to ensure non-negativity.
    #    Then normalize h to sum to 1.
    #    Actually, if we normalize inside the objective, the scale is fixed.
    #    So we can optimize y directly.
    
    def objective(y):
        h = np.exp(y)
        h = h / np.sum(h) # Fix L1 = 1
        return -_score_from_heights(h)
    
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization in log-space
        # Uniform in [0, 1] for h corresponds to (-inf, 0] for log(h) if normalized?
        # Just use random normal or uniform for y.
        y0 = np.random.normal(0, 1, N)
        
        # Use Powell or Nelder-Mead?
        # Powell is derivative-free and often good for this.
        # Nelder-Mead can be slow.
        # Let's try Powell.
        
        try:
            res = minimize(objective, y0, method='Powell',
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_y = res.x
                best_h = np.exp(best_y)
                best_h = best_h / np.sum(best_h)
                
        except Exception:
            continue
        
        if time.time() > deadline:
            break
            
    if best_h is None:
        # Fallback: uniform distribution
        best_h = np.ones(N) / N
    else:
        # Final polish with tighter tolerance?
        # Maybe do one more refinement on the best solution found.
        # But Powell usually converges well.
        # Let's ensure it's non-negative and normalized.
        best_h = np.maximum(best_h, 0)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h / s
        else:
            best_h = np.ones(N) / N
            
    return list(best_h)
