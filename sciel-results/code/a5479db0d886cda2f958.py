import numpy as np
import time
import sys

N = 100

def _max_overlap(h):
    """Compute max over shifts of sum(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        # overlap for shift s: sum of h[i] * inv_h[(i+s)%N]
        # Using np.roll for efficiency
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial solution: Constant vector (Theoretical Optimum)
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = _score(h)
    
    # Strategy: Simulated Annealing / Random Search with Zero-Sum Perturbations
    # We perturb h by adding a random vector p where sum(p) = 0.
    # We project back to [0, 1] and re-normalize to sum N/2.
    
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.95
    
    # Precompute indices for faster roll? No, np.roll is fine for N=100.
    
    step = 0
    while time.time() < deadline:
        # Generate a zero-sum perturbation
        # Random normal, subtract mean to make sum zero
        p = np.random.normal(0, 0.01, N)
        p -= np.mean(p)
        
        h_new = h + p
        
        # Project to [0, 1]
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Re-normalize to sum N/2
        current_sum = h_new.sum()
        if current_sum > 1e-8:
            h_new = h_new * (N/2.0) / current_sum
        else:
            h_new = np.full(N, 0.5)
        
        # Re-project after scaling
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # If sum is not exactly N/2 due to clipping, adjust the largest/smallest elements
        diff = N/2.0 - h_new.sum()
        if abs(diff) > 1e-9:
            if diff > 0:
                # Add to largest elements (least likely to exceed 1)
                idx = np.argsort(h_new)[::-1]
                for i in idx:
                    add = min(diff, 1.0 - h_new[i])
                    if add <= 0: break
                    h_new[i] += add
                    diff -= add
            else:
                # Subtract from smallest elements
                idx = np.argsort(h_new)
                for i in idx:
                    sub = min(-diff, h_new[i])
                    if sub <= 0: break
                    h_new[i] -= sub
                    diff += sub
        
        # Ensure exact sum for the score function's tolerance
        # The score function checks abs(sum - N/2) > 1e-8
        # If we are close, it's fine.
        
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new.copy()
            # Accept better solution
        else:
            # Metropolis acceptance
            delta = val_new - best_val
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                h = h_new.copy()
        
        T *= cooling_rate
        step += 1
        
        # If we have converged to the theoretical minimum, stop early?
        # Theoretical min is N/4.
        if best_val <= N/4.0 + 1e-9:
            break
            
    return best_h
# EVOLVE-BLOCK-END
