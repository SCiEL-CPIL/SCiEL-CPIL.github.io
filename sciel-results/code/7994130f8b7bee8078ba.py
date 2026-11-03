"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines the linear overlap definition (FFT-accelerated) with a two-phase
optimization: (1) SLSQP with smooth-max objective and analytical gradient,
(2) coordinate-descent / subgradient refinement to polish the result.
The structural change is the use of the linear (non-circular) overlap
definition and the smooth-max formulation, which allows the use of
analytical gradients and a well-conditioned optimization landscape.
"""
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
    """Compute the maximum linear overlap."""
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
        if w[k] < 1e-15:
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


def _coordinate_descent(h, max_iter=200, tol=1e-10):
    """
    Coordinate descent refinement: for each coordinate j, find the optimal
    value of h_j given all other coordinates fixed, subject to [0,1] and
    the sum constraint. This is a subgradient method on the smooth max.
    """
    h = _fix_sum(h)
    n = len(h)
    target = n / 2.0
    beta = 50.0
    
    for it in range(max_iter):
        improved = False
        for j in range(n):
            # Compute the current overlap values
            cs = _linear_overlap_values(h)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            
            # For coordinate j, the derivative of C_k w.r.t. h_j is:
            # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k else 0 + h_{j+k} if j+k<n else 0)
            # We want to move h_j in the direction that reduces the weighted max.
            
            # Compute the weighted gradient for coordinate j
            grad_j = 0.0
            for k in range(n):
                if w[k] < 1e-15:
                    continue
                factor = -(2.0 / n) * w[k]
                # Contribution from h_{j-k} if j >= k
                if j >= k:
                    grad_j += factor * h[j - k]
                # Contribution from h_{j+k} if j+k < n
                if j + k < n:
                    grad_j += factor * h[j + k]
            
            # If grad_j > 0, decreasing h_j reduces the objective
            # If grad_j < 0, increasing h_j reduces the objective
            # But we must maintain the sum constraint.
            
            # Try a small step in the direction of -grad_j
            step = 0.01
            if grad_j > 0:
                h_new = h.copy()
                h_new[j] -= step
                # Adjust another coordinate to maintain sum
                # Find the coordinate with the most room to increase
                mask = h_new < 1.0
                mask[j] = False
                if np.any(mask):
                    # Distribute the step among all other coordinates proportionally
                    h_new[mask] += step / np.sum(mask)
                    h_new = np.clip(h_new, 0.0, 1.0)
                    # Re-fix sum
                    h_new = _fix_sum(h_new)
                    
                    score_new = _score(h_new)
                    score_old = _score(h)
                    if score_new < score_old - tol:
                        h = h_new
                        improved = True
            elif grad_j < 0:
                h_new = h.copy()
                h_new[j] += step
                mask = h_new > 0.0
                mask[j] = False
                if np.any(mask):
                    h_new[mask] -= step / np.sum(mask)
                    h_new = np.clip(h_new, 0.0, 1.0)
                    h_new = _fix_sum(h_new)
                    
                    score_new = _score(h_new)
                    score_old = _score(h)
                    if score_new < score_old - tol:
                        h = h_new
                        improved = True
        
        if not improved:
            break
    
    return h


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
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Optimization with Analytical Gradient
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
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
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Refine Cosine Start
    if time.time() - t0 < _DEADLINE - 30:
        _run_optimization(h_cos, beta=50.0, maxiter=200, ftol=1e-9)
        
    # Phase 2: Try other starts if time permits
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8)
            
    # Phase 3: Final Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        _run_optimization(best_h, beta=100.0, maxiter=300, ftol=1e-12)
        
    # Phase 4: Coordinate descent refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_cd = _coordinate_descent(best_h, max_iter=100)
        _update_best(h_cd)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
