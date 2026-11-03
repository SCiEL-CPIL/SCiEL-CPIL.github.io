"""Rewritten program for the second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 50

# Time budget: must finish well within 120s; use 100s as safety margin
START_TIME = time.time()
TIME_LIMIT = 100.0  # seconds

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def _autoconvolution(h):
    """Compute full autoconvolution (linear convolution) of h with itself."""
    return np.convolve(h, h, mode='full')

def _sup_autoconv(h):
    """Supremum norm of the autoconvolution."""
    return np.max(np.abs(_autoconvolution(h)))

def _score(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * sup||autoconv(h)||).
    All heights are nonnegative, so ||h||_1 = sum(h).
    """
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    sup_ac = _sup_autoconv(h)
    if sup_ac <= 0:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _objective_neg_ratio(x):
    """
    Minimize negative of the ratio. x is a vector of nonnegative heights.
    """
    x = np.maximum(x, 0)  # enforce nonnegativity
    l2_sq = np.sum(x ** 2)
    l1 = np.sum(x)
    if l1 <= 0:
        return 0.0
    sup_ac = _sup_autoconv(x)
    if sup_ac <= 0:
        return 0.0
    ratio = l2_sq / (l1 * sup_ac)
    return -ratio

def _soft_sup_autoconv(h, eps=0.01):
    """
    Smooth surrogate for the supremum of the autoconvolution using log-sum-exp.
    This makes the objective differentiable.
    """
    ac = _autoconvolution(h)
    # For nonnegative h, ac is nonnegative, so we can use log-sum-exp directly
    # Use a scaled version to avoid overflow
    max_ac = np.max(ac)
    if max_ac <= 0:
        return 0.0
    # Soft max: max_ac * exp(sum(exp((ac - max_ac)/eps)) / len(ac))
    # But for large arrays, this can be unstable. Use a simpler approach:
    # soft_max = max_ac + eps * log(sum(exp((ac - max_ac)/eps)))
    scaled = (ac - max_ac) / eps
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    log_sum_exp = np.log(np.sum(np.exp(scaled)))
    return max_ac + eps * log_sum_exp

def _objective_neg_ratio_smooth(x, eps=0.01):
    """
    Minimize negative of the ratio with smooth supremum approximation.
    """
    x = np.maximum(x, 0)
    l2_sq = np.sum(x ** 2)
    l1 = np.sum(x)
    if l1 <= 0:
        return 0.0
    sup_ac_soft = _soft_sup_autoconv(x, eps)
    if sup_ac_soft <= 0:
        return 0.0
    ratio = l2_sq / (l1 * sup_ac_soft)
    return -ratio

def _gradient_neg_ratio_smooth(x, eps=0.01):
    """
    Numerical gradient for the smooth objective.
    """
    x = np.maximum(x, 0)
    grad = np.zeros_like(x)
    h = 1e-6
    for i in range(N):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] = max(0, x[i] - h)
        grad[i] = (_objective_neg_ratio_smooth(xp, eps) - _objective_neg_ratio_smooth(xm, eps)) / (2*h)
    return grad

def _construct_initial():
    """Construct an initial guess: uniform vector."""
    return np.ones(N)

def _refine_with_slsqp(h0, max_iter=500, eps=0.01):
    """
    Use SLSQP to refine the solution.
    """
    if _time_left() < 10:
        return h0
    
    # Bounds: each height >= 0
    bounds = [(0, None)] * N
    
    # Objective: minimize negative ratio (smooth version)
    def obj(x):
        return _objective_neg_ratio_smooth(x, eps)
    
    # Use SLSQP with numerical gradient initially, then try analytical
    result = minimize(
        obj,
        h0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-8}
    )
    
    if result.success and result.fun < _objective_neg_ratio_smooth(h0, eps):
        return np.maximum(result.x, 0)
    return h0

def _block_coordinate_descent(h, max_iter=200, eps=0.01):
    """
    Block coordinate descent: optimize pairs of heights around the dominant peak.
    The dominant autoconvolution peak is at lag 0 (sum of squares) or nearby.
    We identify the peak lag and optimize symmetric pairs around it.
    """
    if _time_left() < 5:
        return h
    
    h = np.array(h, dtype=float)
    
    for iteration in range(max_iter):
        if _time_left() < 5:
            break
        
        # Find the lag of the maximum autoconvolution
        ac = _autoconvolution(h)
        peak_lag = np.argmax(ac)
        
        # The autoconvolution at lag k is sum_i h[i] * h[i+k]
        # Peak is typically at lag 0 for nonnegative vectors
        # Optimize pairs symmetric around the center
        
        best_h = h.copy()
        best_score = _score(h)
        
        # Try small perturbations on pairs
        for _ in range(10):  # Limited inner iterations
            # Pick a random pair of indices
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i == j:
                continue
            
            # Try to adjust h[i] and h[j] to improve the ratio
            # Simple line search: try increasing one and decreasing another
            for delta in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                h_trial = h.copy()
                h_trial[i] = max(0, h[i] + delta)
                h_trial[j] = max(0, h[j] - delta)
                score_trial = _score(h_trial)
                if score_trial > best_score:
                    best_score = score_trial
                    best_h = h_trial.copy()
        
        h = best_h
    
    return h

def _polish(h, max_iter=100):
    """
    High-precision polish pass using SLSQP with tighter tolerance.
    """
    if _time_left() < 5:
        return h
    
    bounds = [(0, None)] * N
    
    result = minimize(
        lambda x: _objective_neg_ratio_smooth(x, eps=0.001),
        h,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )
    
    if result.success:
        return np.maximum(result.x, 0)
    return h

def construct():
    """
    Main entry point. Returns a sequence of 50 nonnegative step heights.
    """
    # Start with uniform initialization
    h = _construct_initial()
    best_h = h.copy()
    best_score = _score(h)
    
    # Phase 1: SLSQP refinement with smooth surrogate
    h = _refine_with_slsqp(h, max_iter=300, eps=0.01)
    score = _score(h)
    if score > best_score:
        best_score = score
        best_h = h.copy()
    
    # Phase 2: Block coordinate descent
    if _time_left() > 20:
        h2 = _block_coordinate_descent(best_h, max_iter=100)
        score2 = _score(h2)
        if score2 > best_score:
            best_score = score2
            best_h = h2.copy()
    
    # Phase 3: Polish
    if _time_left() > 10:
        h3 = _polish(best_h, max_iter=100)
        score3 = _score(h3)
        if score3 > best_score:
            best_score = score3
            best_h = h3.copy()
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    # Return as a list
    return list(best_h)
