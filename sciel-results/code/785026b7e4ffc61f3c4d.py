"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def compute_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    # h is a sequence of step heights.
    # "shifted 1-h" means we shift h by k and compute the overlap with 1-h.
    # Overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N])
    # We want to minimize the maximum of these overlaps.
    # Actually, let's re-read the task: "minimize the maximum overlap of h with shifted 1-h"
    # This likely means: for each shift k, compute the dot product of h and (1-h) shifted by k.
    # Let's define overlap(k) = sum_{i=0}^{N-1} h[i] * (1 - h[(i+k) % N])
    # We want to minimize max_k overlap(k).
    
    # Note: sum h[i] = N/2, so sum (1-h[i]) = N/2.
    # The average overlap is (N/2)*(N/2)/N = N/4? No, that's not right.
    # Let's just compute it directly.
    
    max_overlap = 0.0
    for k in range(N):
        shifted_1h = 1 - np.roll(h, -k)
        overlap = np.dot(h, shifted_1h)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different initializations and optimization strategies
    best_h = h0.copy()
    best_score = compute_overlap(h0)
    
    # Strategy 1: Use scipy.optimize.minimize with SLSQP
    # Objective: max overlap
    # Constraints: sum h = N/2, 0 <= h <= 1
    
    def objective(h):
        return compute_overlap(h)
    
    def objective_smooth(h, t=10):
        """Smooth approximation of max overlap using log-sum-exp."""
        overlaps = []
        for k in range(N):
            shifted_1h = 1 - np.roll(h, -k)
            overlap = np.dot(h, shifted_1h)
            overlaps.append(overlap)
        overlaps = np.array(overlaps)
        # Log-sum-exp for smooth max
        m = np.max(overlaps)
        return m + np.log(np.sum(np.exp(overlaps - m)) / N)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2},
    ]
    bounds = [(0, 1)] * N
    
    # Try SLSQP with smooth objective
    try:
        res = minimize(objective_smooth, h0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success and res.fun < best_score:
            best_h = res.x.copy()
            best_score = compute_overlap(best_h)
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best_h
    
    # Try different initializations
    inits = [
        np.full(N, 0.5),
        np.random.uniform(0, 1, N),
        np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)) * 0.5 + 0.5,
        np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)) * 0.5 + 0.5,
    ]
    
    for h0 in inits:
        if time.time() > deadline - 10:
            break
        
        # Normalize to satisfy sum constraint
        h0 = np.clip(h0, 0, 1)
        h0 = h0 / np.sum(h0) * (N/2)
        h0 = np.clip(h0, 0, 1)
        
        try:
            res = minimize(objective_smooth, h0, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
            if res.success:
                score = compute_overlap(res.x)
                if score < best_score:
                    best_h = res.x.copy()
                    best_score = score
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_h, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                score = compute_overlap(res.x)
                if score < best_score:
                    best_h = res.x.copy()
                    best_score = score
        except Exception:
            pass
    
    # Ensure sum constraint is satisfied
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h / np.sum(best_h) * (N/2)
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
