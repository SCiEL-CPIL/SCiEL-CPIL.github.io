"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines analytical gradient optimization with bisection feasibility search.
Strategy: Use bisection to bracket the optimal C_max, then refine with smooth-max SLSQP."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
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
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
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


def _feasibility_check(c_max, h0, maxiter=100):
    """
    Check if there exists h such that max(C_k) <= c_max, sum(h)=N/2, 0<=h_i<=1.
    Returns (feasible, h_solution).
    """
    n = N
    target_sum = n / 2.0
    
    def obj_func(h):
        cs = _linear_overlap_values(h)
        violations = np.maximum(0.0, cs - c_max)
        return np.sum(violations ** 2)
    
    def jac_func(h):
        cs = _linear_overlap_values(h)
        grad = np.zeros(n)
        mask = cs > c_max
        if np.any(mask):
            for k in np.where(mask)[0]:
                w = 2.0 * (cs[k] - c_max)
                factor = -(2.0 / n) * w
                if k > 0:
                    grad[k:] += factor * h[:n-k]
                if k < n:
                    grad[:n-k] += factor * h[k:]
        return grad
    
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum,
             'jac': lambda x: np.ones(n)}]
    bounds = [(0.0, 1.0)] * n
    
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-8})
        h_sol = res.x
        cs_sol = _linear_overlap_values(h_sol)
        return np.max(cs_sol) <= c_max + 1e-6, h_sol
    except Exception:
        return False, h0


def _run_smooth_max_optimization(h0, beta, maxiter, ftol):
    """
    Run SLSQP optimization with smooth max objective.
    Returns optimized h.
    """
    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad
        
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        return res.x
    except Exception:
        return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
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
    
    # 1. Analytical Starts
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # 2. Bisection to bracket optimal C_max
    lo = 0.0
    hi = best_score
    
    # Run bisection for a few iterations to narrow the bracket
    max_bisection_iter = 15
    for it in range(max_bisection_iter):
        if time.time() - t0 > _DEADLINE - 60:
            break
        
        mid = (lo + hi) / 2.0
        
        feasible, h_sol = _feasibility_check(mid, best_h, maxiter=80)
        
        if feasible:
            hi = mid
            _update_best(h_sol)
        else:
            lo = mid
    
    # 3. Refine with smooth max optimization from multiple starting points
    if time.time() - t0 < _DEADLINE - 40:
        _update_best(_run_smooth_max_optimization(best_h, beta=50.0, maxiter=300, ftol=1e-9))
    
    # 4. Try random starts if time permits
    if time.time() - t0 < _DEADLINE - 30:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _update_best(_run_smooth_max_optimization(h_rand, beta=50.0, maxiter=150, ftol=1e-8))
            
    # 5. Final high-precision polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _update_best(_run_smooth_max_optimization(best_h, beta=200.0, maxiter=500, ftol=1e-12))
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
