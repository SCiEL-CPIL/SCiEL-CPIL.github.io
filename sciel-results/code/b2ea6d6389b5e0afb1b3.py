"""
Minimize max|autoconv| / (sum(x))^2 using continuous optimization.
Strategy:
1. Use L2 norm of autoconvolution as a proxy for flat spectrum (fast analytic gradient).
2. Refine using local search on the continuous variables to minimize the actual max-abs metric.
"""
import numpy as np
import time
from scipy.signal import fftconvolve
from scipy.optimize import minimize

N = 400

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = fftconvolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _compute_l2_energy(x):
    """Compute sum of squares of autoconvolution (L2 norm squared)."""
    c = fftconvolve(x, x, mode='full')
    return np.sum(c**2)

def _local_search_continuous(x, deadline, max_rounds=5):
    """
    Local search on continuous variables.
    For each index, try a few multiplicative perturbations.
    """
    current = x.copy()
    current_score = _compute_score(current)
    
    for r in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            orig_val = current[i]
            # Try a few perturbations
            for factor in [0.8, 0.9, 1.1, 1.2]:
                new_val = orig_val * factor
                current[i] = new_val
                new_score = _compute_score(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = orig_val
        if not improved:
            break
            
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. L2 Minimization to get a flat-spectrum-like sequence
    # We minimize the L2 energy of the convolution.
    # This is equivalent to flattening the magnitude spectrum of x.
    
    best_x = None
    best_score = float('inf')
    
    # Initializations
    inits = [
        np.ones(N),
        np.random.RandomState(42).rand(N),
        np.random.RandomState(123).rand(N) + 0.1
    ]
    
    for x0 in inits:
        if time.time() > deadline - 20:
            break
            
        # Normalize x0 to sum to 1 to avoid scaling issues during optimization
        x0_norm = x0 / np.sum(x0)
        
        # Minimize L2 energy
        # L-BFGS-B with analytic gradient is fast.
        # Gradient of sum(c^2) where c = x*x.
        # We can just let scipy compute it or use a simple gradient.
        # For speed, we'll use 'L-BFGS-B' with default numerical gradient if analytic is complex.
        # But N=400 is small enough for numerical gradient in L2 if iterations are few.
        
        try:
            res = minimize(_compute_l2_energy, x0_norm, method='L-BFGS-B', 
                           options={'maxiter': 20, 'gtol': 1e-4})
            x_opt = res.x
            # Restore scale? The metric is scale invariant, so we can use x_opt directly.
            # But we might want to ensure sum is positive and large.
            # x_opt should be fine.
            
            score = _compute_score(x_opt)
            if score < best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            continue

    if best_x is None:
        best_x = np.ones(N)
        best_score = _compute_score(best_x)
    
    # 2. Local Search Refinement
    if time.time() < deadline - 5:
        best_x, best_score = _local_search_continuous(best_x, deadline - 5, max_rounds=10)
        
    return best_x
