"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial guess: uniform 0.5
    x0 = np.full(N, 0.5)
    
    # Compute overlap values for a given h
    def compute_overlaps(h):
        # h has length N, represents step heights on [0, 2] with N steps of width 2/N
        # 1-h is the complement
        # Overlap at shift k: sum_i h[i] * (1 - h[(i-k) mod N]) for cyclic shifts
        # But we need to be careful about boundary - the problem says [0,2]
        # Let's use cyclic convolution for simplicity, or linear with zero padding
        
        # Using linear overlap (zero-padding outside [0, N-1])
        # For shift k, overlap_k = sum_{i} h[i] * (1-h[i-k]) where i-k is in [0, N-1]
        # Actually, let me think of it as: shift (1-h) by k positions, then element-wise multiply with h and sum
        
        one_minus_h = 1.0 - h
        overlaps = []
        for k in range(-N + 1, N):
            # Shift one_minus_h by k: result[i] = one_minus_h[i - k] if 0 <= i-k < N else 0
            # overlap = sum_i h[i] * shifted_omh[i]
            # This is equivalent to cross-correlation
            if k >= 0:
                # shifted_omh[i] = one_minus_h[i - k] for i >= k, else 0
                # overlap = sum_{i=k}^{N-1} h[i] * one_minus_h[i - k]
                valid_i = range(k, N)
                overlap = np.sum(h[valid_i] * one_minus_h[np.arange(N - k)])
            else:
                # k < 0, let m = -k > 0
                # shifted_omh[i] = one_minus_h[i - k] = one_minus_h[i + m] for i < N - m, else 0
                m = -k
                valid_i = range(N - m)
                overlap = np.sum(h[valid_i] * one_minus_h[np.arange(m, N)])
            overlaps.append(overlap)
        return np.array(overlaps)
    
    def objective(x):
        # Soft-max of overlaps to make it differentiable
        overlaps = compute_overlaps(x)
        # Use log-sum-exp for soft max
        max_val = np.max(overlaps)
        if max_val > 1e10:
            return max_val
        log_sum = np.log(np.sum(np.exp(overlaps - max_val))) + max_val
        return log_sum
    
    def constraints(x):
        # Sum constraint: sum(x) = N/2
        # Bounds: 0 <= x <= 1 (handled by bounds in minimize)
        return N / 2.0 - np.sum(x)
    
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': constraints}]
    
    best_x = x0.copy()
    best_score = compute_overlaps(best_x).max()
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         constraints=cons, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or np.sum(np.abs(result.x - 0.5)) > 1e-6:
            # Check if constraint is satisfied
            if abs(np.sum(result.x) - N / 2.0) < 1e-4:
                # Clip to [0,1]
                result.x = np.clip(result.x, 0, 1)
                # Renormalize to satisfy sum = N/2
                diff = N / 2.0 - np.sum(result.x)
                result.x += diff / N
                result.x = np.clip(result.x, 0, 1)
                
                score = compute_overlaps(result.x).max()
                if score < best_score:
                    best_score = score
                    best_x = result.x.copy()
    except Exception:
        pass
    
    # Coordinate descent polish on active set
    if time.time() < deadline:
        # Do coordinate descent to refine
        x = best_x.copy()
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                old_val = x[i]
                # Try small perturbations
                for delta in [-0.01, -0.001, 0.001, 0.01, -0.005, 0.005]:
                    new_val = old_val + delta
                    if new_val < 0 or new_val > 1:
                        continue
                    x[i] = new_val
                    # Adjust another coordinate to maintain sum
                    # Find j != i to adjust
                    j = (i + 1) % N
                    if j == i:
                        j = (i + 2) % N
                    adj = old_val - new_val
                    x[j] = x[j] + adj
                    if x[j] < 0 or x[j] > 1:
                        x[i] = old_val
                        x[j] = x[j] - adj
                        continue
                    score = compute_overlaps(x).max()
                    if score < best_score - 1e-8:
                        best_score = score
                        best_x = x.copy()
                        improved = True
                        break
                    else:
                        x[i] = old_val
                        x[j] = x[j] - adj
                if not improved and i > 50:
                    break
    
    # Ensure exact sum
    diff = N / 2.0 - np.sum(best_x)
    best_x += diff / N
    best_x = np.clip(best_x, 0, 1)
    
    return best_x

# EVOLVE-BLOCK-END
