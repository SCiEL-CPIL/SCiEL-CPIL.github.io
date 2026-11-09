"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr_max(signed_steps):
    """Compute the maximum absolute autocorrelation at non-zero integer lags.
    
    signed_steps: array of N values, each ±1, representing step heights on N equal intervals.
    The domain is [-1/4, 1/4] with N intervals, so interval width = 1/(2N) = 1/800.
    
    The autocorrelation at integer lag k (in units of interval width) is:
    R(k) = sum_{i} f[i] * f[i+k] * (interval_width)
    but for the "upper bound" we likely care about the normalized or raw sum.
    
    Actually, for a step function, the autoconvolution at offset h (in domain units) 
    is the integral. For integer lags in interval units, the overlap is exact.
    
    Let's compute the sum S(k) = sum_{i=0}^{N-1-k} f[i]*f[i+k] for k=1..N-1,
    and also for negative lags by symmetry. The maximum |S(k)| is our target.
    """
    n = len(signed_steps)
    f = signed_steps
    max_abs = 0.0
    for k in range(1, n):
        # S(k) = sum_{i=0}^{n-1-k} f[i]*f[i+k]
        s = np.dot(f[:n-k], f[k:])
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = abs_s
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    best_score = float('inf')
    best_steps = np.ones(N, dtype=np.float64)
    
    # Start with a random ±1 pattern
    np.random.seed(42)
    best_steps = np.random.choice([-1.0, 1.0], size=N)
    best_score = _compute_autocorr_max(best_steps)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_steps
    
    # Try to improve via local search: flip individual signs
    # Use a greedy/local search approach
    current = best_steps.copy()
    current_score = best_score
    
    # Multiple restarts with random initializations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        current = np.random.choice([-1.0, 1.0], size=N)
        current_score = _compute_autocorr_max(current)
        
        if current_score < best_score:
            best_score = current_score
            best_steps = current.copy()
        
        # Local search: try flipping each position
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try flipping position i
                old_val = current[i]
                current[i] = -old_val
                new_score = _compute_autocorr_max(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_steps = current.copy()
                else:
                    current[i] = old_val
    
    return best_steps
# EVOLVE-BLOCK-END
