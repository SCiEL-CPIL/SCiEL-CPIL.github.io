"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Differential Evolution (global stochastic search) combined with a local
polish using Nesterov Accelerated Projected Gradient Descent (PGD).
This is a structural change from the current local-first approach (SLSQP/PGD with
bisection logic). Differential Evolution explores the global landscape using
populations of candidates, which helps escape local minima inherent in the
non-convex max-overlap objective. The local polish refines the best candidate
found by DE using a fast, robust first-order method.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
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

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _project(h):
    """
    Project h onto the set {x : sum x = N/2, 0 <= x <= 1}.
    """
    h = np.asarray(h, dtype=np.float64)
    target = N / 2.0
    
    # If sum is already close and within bounds, return
    if np.all(h >= 0) and np.all(h <= 1) and abs(np.sum(h) - target) < 1e-9:
        return h
    
    # Binary search for lambda such that sum(clip(h - lambda, 0, 1)) = target
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target:
            lo = mid
        else:
            hi = mid
            
    proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    # Fine tune to ensure exact sum
    diff = target - np.sum(proj)
    if abs(diff) > 1e-12:
        mask = (proj > 0) & (proj < 1)
        if np.any(mask):
            proj[mask] += diff / np.sum(mask)
            proj = np.clip(proj, 0.0, 1.0)
    return proj

def _smooth_max_grad(h, beta):
    """
    Compute value and gradient of smooth max approximation of C_k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk == 0.0:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
    return val, grad

def _nesterov_pgd(h0, max_iter=500, step_size=0.05, beta=100.0):
    """
    Minimize smooth max of C_k using Nesterov Accelerated PGD.
    """
    h = h0.copy()
    h_prev = h0.copy()
    t_k = 1.0
    
    for i in range(max_iter):
        t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        y = h + (t_k - 1) / t_new * (h - h_prev)
        
        val, grad = _smooth_max_grad(y, beta)
        
        # Projected Gradient Step
        h_new = _project(y - step_size * grad)
        
        h_prev = h
        h = h_new
        t_k = t_k # Note: standard Nesterov uses t_new, but for stability we can stick to t_k or update.
        t_k = t_new
        
        # Early stopping
        if i > 100 and abs(val - _score(h)) < 1e-6:
             # Check if gradient is small? 
             pass

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
            
    # Initial guess: Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Define objective for Differential Evolution
    # DE works on unconstrained variables in [0, 1].
    # We will map the DE output to satisfy sum constraint?
    # Actually, DE in scipy handles bounds. But sum constraint is tricky.
    # We can penalize the sum constraint in the objective.
    # Or, better: DE on N-1 variables, fix the last one?
    # Or: DE on N variables, but project the result to the feasible set before scoring.
    # If we project, the objective landscape changes.
    # Let's try: DE minimizes max_overlap(project(x)).
    
    def de_objective(x):
        # x is in [0, 1]^N
        h_proj = _project(x)
        return _score(h_proj)
    
    # Run Differential Evolution
    # popsize=15 is default, maxiter=1000 might be too much.
    # We have ~100s left.
    # Let's run DE for a fixed number of iterations or time.
    
    try:
        # Use a custom callback to check time
        def callback(xk, convergence):
            return time.time() - t0 > _DEADLINE - 20
        
        res = differential_evolution(
            de_objective,
            bounds=[(0, 1)] * N,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-4,
            callback=callback,
            polish=False  # We do our own polish
        )
        
        if res.x is not None:
            h_de = _project(res.x)
            _update_best(h_de)
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Polish the best candidate found using Nesterov PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        # Use a few restarts for polish
        for restart in range(3):
            if time.time() - t0 > _DEADLINE - 2:
                break
            
            if restart == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.05 * np.random.randn(N)
                h_init = _fix_sum(h_init)
            
            h_polished = _nesterov_pgd(h_init, max_iter=200, step_size=0.05, beta=100.0)
            _update_best(h_polished)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
