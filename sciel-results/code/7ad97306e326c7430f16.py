import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap(h):
    """Compute max over shifts of mean(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        # overlap for shift s: mean of h[i] * inv_h[(i+s)%N]
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
    deadline = start_time + 100  # Safety margin
    
    # The theoretical optimum is the constant sequence h_i = 0.5
    h_const = np.full(N, 0.5)
    best_h = h_const.copy()
    best_val = _score(h_const)
    
    # Strategy: Multi-stage optimization
    # Stage 1: Binary Local Search to find a good discrete pattern
    # Stage 2: Continuous Refinement using Nelder-Mead
    
    # Initialize with a random binary sequence with N/2 ones
    def random_binary():
        h = np.zeros(N)
        ones = np.random.choice(N, N//2, replace=False)
        h[ones] = 1.0
        return h
    
    # Try a few random binary starts
    for _ in range(5):
        if time.time() > deadline:
            break
        h_bin = random_binary()
        
        # Local search on binary: flip pairs to maintain sum
        # Objective: minimize max overlap
        current_val = _score(h_bin)
        
        # Simple local search: try flipping a 1 to 0 and a 0 to 1
        # To speed up, we can just evaluate a few random moves
        for _ in range(100):
            if time.time() > deadline:
                break
            # Pick a random 1 and a random 0
            ones_idx = np.where(h_bin == 1.0)[0]
            zeros_idx = np.where(h_bin == 0.0)[0]
            if len(ones_idx) == 0 or len(zeros_idx) == 0:
                break
            i1 = np.random.choice(ones_idx)
            i0 = np.random.choice(zeros_idx)
            
            h_new = h_bin.copy()
            h_new[i1] = 0.0
            h_new[i0] = 1.0
            
            new_val = _score(h_new)
            if new_val < current_val:
                h_bin = h_new
                current_val = new_val
        
        # Stage 2: Refine the best binary pattern found so far
        if current_val < best_val:
            best_h = h_bin.copy()
            best_val = current_val
        
        # Use Nelder-Mead to refine the binary pattern into a real-valued one
        # We need to handle the constraint sum(h) = N/2 and 0 <= h <= 1
        # Nelder-Mead doesn't handle constraints well, so we project or use a penalty
        # Alternatively, we can just use the binary pattern as a starting point for a constrained optimizer
        # But given time limits, a simple gradient step or Nelder-Mead on a penalized objective might be better
        
        # Let's try Nelder-Mead on the objective with a penalty for constraint violation
        def objective(h_flat):
            h = h_flat.reshape(N)
            # Penalty for sum
            pen_sum = (np.sum(h) - N/2.0)**2 * 1000
            # Penalty for bounds
            pen_bound = np.sum(np.maximum(0, -h)**2 + np.maximum(0, h-1)**2) * 1000
            return _max_overlap(h) + pen_sum + pen_bound
        
        # Start from the binary pattern
        x0 = h_bin.copy()
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if time.time() > deadline:
                break
            h_opt = res.x.reshape(N)
            # Normalize to satisfy sum constraint exactly
            if np.sum(h_opt) > 0:
                h_opt = h_opt / np.sum(h_opt) * (N/2.0)
            # Clip to bounds
            h_opt = np.clip(h_opt, 0, 1)
            # Re-normalize after clipping
            diff = N/2.0 - np.sum(h_opt)
            if diff > 0:
                # Add to smallest elements
                idx = np.argsort(h_opt)
                for i in idx:
                    if diff <= 0: break
                    add = min(diff, 1.0 - h_opt[i])
                    h_opt[i] += add
                    diff -= add
            elif diff < 0:
                idx = np.argsort(h_opt)[::-1]
                for i in idx:
                    if diff >= 0: break
                    sub = min(-diff, h_opt[i])
                    h_opt[i] -= sub
                    diff += sub
            
            val_opt = _score(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
        except Exception:
            pass

    # Final check: ensure the constant sequence is considered
    # It is already in best_h if it was the best found, or we can explicitly compare
    val_const = _score(h_const)
    if val_const < best_val:
        best_h = h_const
        best_val = val_const

    # Ensure valid output
    assert abs(best_h.sum() - N/2.0) < 1e-8
    assert best_h.min() >= -1e-8 and best_h.max() <= 1.0 + 1e-8
    
    return best_h
# EVOLVE-BLOCK-END
