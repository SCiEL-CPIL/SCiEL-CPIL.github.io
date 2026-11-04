import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0

def _raw_score(heights):
    """Compute the raw value to be minimized."""
    dx = 1.0 / (N * 2)
    s = np.sum(heights)
    if s <= 0:
        return float('inf')
    conv = np.convolve(heights, heights, mode='full') * dx
    return np.max(conv) / (s * dx)**2

def construct():
    start_time = time.time()
    
    # Uniform baseline
    uniform = np.ones(N)
    best_heights = uniform.copy()
    best_score = _raw_score(uniform)
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    dx = 1.0 / (N * 2)
    
    # Strategy: Local search on binary/bang-bang structures
    # Start with several boxcar configurations
    
    candidates = []
    
    # 1. Uniform
    candidates.append(uniform.copy())
    
    # 2. Single Boxcars of various lengths
    for L in [100, 200, 300, 400, 500, 600]:
        h = np.zeros(N)
        h[:L] = 1.0
        candidates.append(h)
        
    # 3. Two Boxcars
    for L1 in [100, 200, 300]:
        for L2 in [100, 200, 300]:
            if L1 + L2 <= N:
                h = np.zeros(N)
                h[:L1] = 1.0
                h[N-L2:] = 1.0
                candidates.append(h)
                
    # 4. Random Bang-Bang (Random subsets of intervals)
    np.random.seed(42)
    for _ in range(20):
        h = np.zeros(N)
        # Randomly choose intervals to turn on
        num_intervals = np.random.randint(1, 10)
        for _ in range(num_intervals):
            start = np.random.randint(0, N)
            length = np.random.randint(10, N - start)
            h[start:start+length] = 1.0
        candidates.append(h)
        
    # Evaluate all candidates
    for h in candidates:
        s = _raw_score(h)
        if s < best_score:
            best_score = s
            best_heights = h.copy()
            
    # Local Search: Simulated Annealing / Hill Climbing on the best candidate
    # We will operate on the heights directly, allowing small perturbations
    # and "flips" to 0 or 1.
    
    h = best_heights.copy()
    current_score = best_score
    
    # Normalize to sum = N to keep scale consistent? 
    # The score is scale invariant: score(c*h) = score(h).
    # So we can fix sum(h) = N.
    
    # Define a move: pick a random index i, change h[i] to a new value.
    # Or pick a random interval and shift it.
    
    # Let's use a simple random walk with cooling.
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    while check_time() and T > T_min:
        # Generate a move
        move_type = np.random.rand()
        
        if move_type < 0.5:
            # Flip a random bit (0->1 or 1->0)
            i = np.random.randint(0, N)
            old_val = h[i]
            h[i] = 1.0 if old_val < 0.5 else 0.0
            new_score = _raw_score(h)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_heights = h.copy()
            else:
                h[i] = old_val # Revert
        else:
            # Shift a random interval
            # Find a contiguous block of 1s or 0s
            # This is complex, so let's just do a small random perturbation of a random segment
            i = np.random.randint(0, N - 10)
            length = np.random.randint(5, 20)
            old_segment = h[i:i+length].copy()
            # Shift the segment right by 1
            h[i:i+length] = 0.0
            if i + length < N:
                h[i+1:i+length+1] = old_segment
            new_score = _raw_score(h)
            
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_heights = h.copy()
            else:
                h[i:i+length] = old_segment # Revert
                if i + length < N:
                    h[i+1:i+length+1] = 0.0
            
        T *= alpha
        
    # Final polish with gradient descent on the smooth surrogate
    if check_time():
        # Use the best bang-bang solution as initialization for a smooth optimization
        # This might help if the optimal solution is not strictly binary.
        
        def soft_max(conv, tau=0.01):
            m = np.max(conv)
            return m + tau * np.log(np.sum(np.exp((conv - m) / tau)))
        
        def objective_smooth(h, tau=0.01):
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s <= 0:
                return 1e10
            conv = np.convolve(h, h, mode='full') * dx
            sm = soft_max(conv, tau)
            return sm / (s * dx)**2
            
        # Initialize with best_heights
        x0 = best_heights.copy()
        
        try:
            result = minimize(objective_smooth, x0, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = np.maximum(result.x, 0)
            s_opt = _raw_score(h_opt)
            if s_opt < best_score:
                best_score = s_opt
                best_heights = h_opt.copy()
        except Exception:
            pass

    return list(best_heights)
