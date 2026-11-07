"""Erdos minimum overlap - Discrete Permutation Search + Continuous Polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    """Compute the exact value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    # Final fine-tuning for precision
    s = h.sum()
    if abs(s - target) > 1e-10:
        diff = target - s
        # Adjust free elements
        free = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def _objective_and_grad(x):
    """Objective: smooth max of correlation terms (log-sum-exp)."""
    h = x
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    
    # Smooth max via log-sum-exp
    beta = 50.0
    c = corr * 2.0 / N
    c_shifted = c - np.max(c)
    soft_max = np.max(c) + np.log(np.sum(np.exp(beta * c_shifted))) / beta
    
    # Penalty for bounds (L-BFGS-B handles bounds, but this helps if it drifts)
    # Actually, L-BFGS-B with bounds is robust. We can skip penalties if we trust bounds.
    # But sum constraint is not handled by bounds.
    # We can add a penalty for sum != N/2.
    sum_h = np.sum(h)
    sum_penalty = 100.0 * (sum_h - N / 2.0) ** 2
    
    total = soft_max + sum_penalty
    
    # Gradient
    weights = np.exp(beta * c_shifted)
    weights = weights / np.sum(weights)
    grad_corr = weights * 2.0 / N
    
    grad_h = np.zeros(N)
    for k in range(2 * N - 1):
        gk = grad_corr[k]
        if gk == 0:
            continue
        if k < N:
            for j in range(N - 1 - k):
                grad_h[j+k] += gk * (1.0 - h[j])
                grad_h[j] += gk * (-h[j+k])
        else:
            m = k - N + 1
            for j in range(m, N):
                grad_h[j] += gk * (1.0 - h[j-m])
                grad_h[j-m] += gk * (-h[j])
                
    grad_h += 200.0 * (sum_h - N / 2.0)
    
    return total, grad_h

def local_search_discrete(h_init, max_epochs=50, time_limit=5.0):
    """Perform local search by swapping elements in the fixed multiset."""
    h = h_init.copy()
    current_val = compute_value(h)
    start_time = time.time()
    
    improved = True
    epoch = 0
    while improved and (time.time() - start_time) < time_limit and epoch < max_epochs:
        improved = False
        epoch += 1
        # Randomize order of pairs to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            if (time.time() - start_time) > time_limit:
                break
            for j in indices[i+1:]:
                if i == j:
                    continue
                # Swap
                h[i], h[j] = h[j], h[i]
                new_val = compute_value(h)
                if new_val < current_val - 1e-12:
                    current_val = new_val
                    improved = True
                else:
                    # Revert
                    h[i], h[j] = h[j], h[i]
                    
    return h, current_val

def construct():
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Define multisets (must sum to 50)
    # Symmetric around 0.5
    multisets = [
        [0.0] * 50 + [1.0] * 50,
        [0.25] * 50 + [0.75] * 50,
        [0.1] * 25 + [0.5] * 50 + [0.9] * 25,
        [0.0] * 25 + [0.5] * 50 + [1.0] * 25,
        [0.2] * 25 + [0.4] * 25 + [0.6] * 25 + [0.8] * 25, # Sum = 25*(2.0) = 50
        [0.1] * 20 + [0.3] * 20 + [0.5] * 20 + [0.7] * 20 + [0.9] * 20, # Sum = 20*(2.5) = 50
    ]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Discrete Search
    for ms in multisets:
        if remaining() < 10.0:
            break
            
        # Run a few restarts for this multiset
        for restart in range(5):
            if remaining() < 5.0:
                break
                
            # Random permutation
            h0 = np.array(ms, dtype=float)
            rng.shuffle(h0)
            
            # Local search
            h_opt, val = local_search_discrete(h0, max_epochs=100, time_limit=3.0)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
    # Phase 2: Continuous Polish
    # Use the best discrete solution as a starting point for L-BFGS-B
    if remaining() > 10.0:
        try:
            h_start = best_h.copy()
            # Ensure it's a valid starting point for the continuous optimizer
            h_start = project(h_start)
            
            res = minimize(
                _objective_and_grad,
                h_start,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_polished = np.clip(res.x, 0, 1)
            h_polished = project(h_polished)
            
            val = compute_value(h_polished)
            if val < best_val:
                best_val = val
                best_h = h_polished.copy()
                
        except Exception:
            pass

    # Final verification
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
        
    # Ensure sum is exactly N/2 within tolerance
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
