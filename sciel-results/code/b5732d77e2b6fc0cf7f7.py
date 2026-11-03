"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Multiple restarts of SLSQP optimization on a smoothed (log-sum-exp)
max overlap objective. SLSQP handles the equality constraint and bounds directly.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115


def _compute_overlaps(h):
    """Compute C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_overlaps(h)
    return float(np.max(cs))


def _project_to_feasible(y):
    """
    Project vector y onto the set {h | 0 <= h_i <= 1, sum(h) = N/2}.
    Uses bisection on the threshold tau.
    """
    n = len(y)
    target_sum = n / 2.0
    
    lo = np.min(y) - 2.0
    hi = np.max(y) + 2.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h = np.clip(y - lo, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def _soft_max_obj_and_grad(h, alpha):
    """
    Compute log-sum-exp of overlaps and its gradient.
    Objective: (1/alpha) * log(sum(exp(alpha * C_k)))
    """
    cs = _compute_overlaps(h)
    
    c_max = np.max(cs)
    exp_cs = np.exp(alpha * (cs - c_max))
    sum_exp = np.sum(exp_cs)
    
    obj = (1.0 / alpha) * (np.log(sum_exp) + alpha * c_max)
    
    grad = np.zeros(N)
    w = exp_cs / sum_exp
    
    grad_raw = np.zeros(N)
    for j in range(N):
        part1 = np.dot(w[:j+1], h[j::-1])
        part2 = np.dot(w[:N-j], h[j:])
        grad_raw[j] = -(2.0 / N) * (part1 + part2)
        
    grad = grad_raw
    return obj, grad


def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses multiple restarts of SLSQP optimization.
    """
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # We will run SLSQP multiple times with different initializations.
    # Each run uses a specific alpha for smoothing.
    # We can vary alpha or just use a fixed high alpha for sharpness.
    # SLSQP works well with smooth functions.
    
    num_restarts = 50 # Upper bound, will stop based on time
    
    for i in range(num_restarts):
        if time.time() > _DEADLINE - 5:
            break
            
        # Generate a random feasible starting point
        # Strategy: Mix of random and structured
        if i % 2 == 0:
            # Random
            h0_raw = np.random.uniform(0, 1, N)
        else:
            # Structured: sinusoidal perturbations
            freq = np.random.randint(1, N//2)
            phase = np.random.uniform(0, 2*np.pi)
            h0_raw = 0.5 + 0.3 * np.cos(2.0 * np.pi * freq * np.arange(N) / N + phase)
            
        h0 = _project_to_feasible(h0_raw)
        
        # Define objective and gradient for SLSQP
        # Use a high alpha to approximate max closely, but keep it smooth enough for SLSQP
        # Alpha=50-100 is usually good.
        alpha = 50.0 
        
        def fun(x):
            val, _ = _soft_max_obj_and_grad(x, alpha)
            return val
            
        def jac(x):
            _, g = _soft_max_obj_and_grad(x, alpha)
            return g
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0})
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(fun, h0, method='SLSQP', jac=jac, bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success and res.fun < best_score:
                # Verify the actual score (not smoothed)
                actual_score = _score(res.x)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = res.x.copy()
                    
        except Exception:
            pass
            
    if best_h is None:
        # Fallback to a simple feasible solution
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure constraints are met exactly
    best_h = _project_to_feasible(best_h)
    
    return best_h
