"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines analytical cosine structure with efficient linear-overlap gradient optimization.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 10.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    This corresponds to the integral of h(x) * (1 - h(x-k)) over the domain,
    assuming h is zero outside [0, N].
    """
    n = len(h)
    cs = np.zeros(n)
    # k=0
    r0 = np.dot(h, h)
    cs[0] = (2.0 / n) * (n / 2.0 - r0)
    
    for k in range(1, n):
        # r_k = sum_{i=k}^{n-1} h[i] * h[i-k]
        # This is equivalent to np.dot(h[k:], h[:-k])
        r = np.dot(h[k:], h[:-k])
        cs[k] = (2.0 / n) * (n / 2.0 - r)
        
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Efficient gradient calculation:
    # grad[j] = sum_k w[k] * (-(2/n) * (h[j-k] + h[j+k]))
    # We can compute this using convolutions or simple vectorized sums.
    
    # Let's use a vectorized approach.
    # For each k, we need to add w[k] * (-(2/n)) * (shifted_h_down + shifted_h_up) to grad.
    
    factor = -(2.0 / n)
    
    # We iterate over k, but we can vectorize the inner part if we precompute shifts?
    # Since N=100, a loop over k is fine, but let's make the inner op vectorized.
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        # Contribution to grad from k-th overlap term
        # dC_k/dh_j involves h[j-k] and h[j+k]
        
        # Term 1: h[j-k] for j >= k
        # This is h[0:N-k] added to grad[k:N]
        if k < n:
            grad[k:] += w[k] * factor * h[:n-k]
            
        # Term 2: h[j+k] for j+k < n => j < n-k
        # This is h[k:N] added to grad[0:N-k]
        if k < n:
            grad[:n-k] += w[k] * factor * h[k:]
            
    return val, grad

def _fix_sum(h, target=None):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    if target is None:
        target = TARGET_SUM
    h = np.array(h, dtype=np.float64, copy=True)
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _run_optimization(h0, beta, maxiter, ftol, deadline):
    """
    Run SLSQP optimization with the smooth max objective and its gradient.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < float('inf'):
            h_new = _fix_sum(res.x)
            return h_new, _score(h_new)
    except Exception:
        pass
    return None, float('inf')

def _local_polish(h, deadline):
    """
    Perform local coordinate descent and pair swaps to polish the solution.
    """
    h = _fix_sum(h.copy())
    current_score = _score(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Coordinate descent with small steps
        step_sizes = [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]
        for i in range(N):
            if time.time() > deadline:
                break
            for ds in step_sizes:
                h_trial = h.copy()
                h_trial[i] += ds
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s_trial = _score(h_trial)
                if s_trial < current_score - 1e-12:
                    h = h_trial
                    current_score = s_trial
                    improved = True
                    break
        
        # 2. Pair swaps (to handle permutation invariance or local minima)
        if not improved:
            for i in range(N - 1):
                if time.time() > deadline:
                    break
                h_trial = h.copy()
                h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
                h_trial = _fix_sum(h_trial)
                s_trial = _score(h_trial)
                if s_trial < current_score - 1e-12:
                    h = h_trial
                    current_score = s_trial
                    improved = True
                    break
                    
    return h, current_score

def construct():
    t0 = time.time()
    deadline = t0 + (150.0 - DEADLINE_MARGIN)
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Start: Cosine Wave
    # h_i = 0.5 + 0.5 * cos(2 * pi * i / N)
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Optimization with Analytical Gradient
    # Phase 1: Refine Cosine Start with moderate beta
    if time.time() < deadline - 30:
        h_opt, s_opt = _run_optimization(h_cos, beta=50.0, maxiter=200, ftol=1e-9, deadline=deadline)
        if h_opt is not None:
            _update_best(h_opt)
    
    # Phase 2: Try other starts if time permits
    # Random starts
    if time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            h_opt, s_opt = _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8, deadline=deadline)
            if h_opt is not None:
                _update_best(h_opt)
                
    # Phase 3: Final Polish with higher beta
    if best_h is not None and time.time() < deadline - 5:
        h_opt, s_opt = _run_optimization(best_h, beta=100.0, maxiter=300, ftol=1e-12, deadline=deadline)
        if h_opt is not None:
            _update_best(h_opt)
            
    # Phase 4: Local Polish
    if best_h is not None and time.time() < deadline - 1:
        h_polished, s_polished = _local_polish(best_h, deadline)
        _update_best(h_polished)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
