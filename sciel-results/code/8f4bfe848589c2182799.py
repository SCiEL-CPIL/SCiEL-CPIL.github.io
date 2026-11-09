"""Strategy: Binary sequence local search with restarts to minimize peak sidelobe of autocorrelation."""
import numpy as np
import time
from scipy.fft import fft, ifft

N = 400

# EVOLVE-BLOCK-START
def _autocorr_full(x):
    """Compute full linear autocorrelation via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft(x, size)
    ac = ifft(X * np.conj(X)).real[:2 * n - 1]
    return ac

def _objective(x):
    """
    Objective: max |AC(k)| for k != 0, divided by (sum x)^2.
    Lower is better. We exclude lag 0 (the peak).
    """
    s = np.sum(x)
    if s == 0:
        return float('inf')
    ac = _autocorr_full(x)
    # In linear autocorrelation via FFT (zero-padded), the lag 0 term is at index N-1.
    # We exclude it.
    ac_no_lag0 = np.delete(ac, N - 1)
    max_val = np.max(np.abs(ac_no_lag0))
    return max_val / (s * s)

def _flip_bit(x, idx):
    """Flip the sign of a single bit (1 -> -1 or -1 -> 1)."""
    new_x = x.copy()
    new_x[idx] *= -1
    return new_x

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s margin for safety.
    deadline = start_time + 140.0
    
    # Best result so far
    # Start with all +1 (constant sequence), which is a known good baseline.
    best_x = np.ones(N, dtype=np.float64)
    best_score = _objective(best_x)
    
    # We will perform multiple restarts.
    # Each restart starts from a random binary sequence or a perturbed version of the best.
    
    max_restarts = 50 # Limit restarts to ensure we don't run out of time
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize current solution for this restart
        # Option 1: Start from best_x
        # Option 2: Start from random
        # Option 3: Start from best_x with some random flips
        if r == 0:
            x = best_x.copy()
        elif r % 3 == 0:
            x = np.ones(N, dtype=np.float64)
            x *= np.random.choice([-1, 1], N)
        elif r % 3 == 1:
            x = best_x.copy()
            num_flips = np.random.randint(5, 20)
            indices = np.random.choice(N, num_flips, replace=False)
            x[indices] *= -1
        else:
            # Start from a random walk? Or just random.
            x = np.ones(N, dtype=np.float64)
            x *= np.random.choice([-1, 1], N)
            
        current_score = _objective(x)
        
        # If this start is better than global best, update
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Local Search: Simulated Annealing style on binary states
        # Temperature starts high, cools down.
        T = 0.5
        cooling = 0.995
        steps_per_restart = 10000 # Adjust based on time? Let's do dynamic steps.
        
        # Dynamic steps based on remaining time
        # Estimate time per step: autocorr is O(N log N) ~ 400*9 ~ 3600 ops. Very fast.
        # 10000 steps * 3600 ops = 3.6e7 ops. Should be < 1s.
        # So we can do many steps.
        
        while time.time() < deadline - 5.0:
            # Generate neighbor: flip 1 or 2 bits
            num_flips = 1 if np.random.random() < 0.8 else 2
            indices = np.random.choice(N, num_flips, replace=False)
            
            x_new = x.copy()
            x_new[indices] *= -1
            
            new_score = _objective(x_new)
            
            if new_score < current_score:
                x = x_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                delta_e = new_score - current_score
                if T > 1e-6:
                    prob = np.exp(-delta_e / T)
                    if np.random.random() < prob:
                        x = x_new
                        current_score = new_score
            
            T *= cooling
            
            # If T is very small, break to save time for next restart?
            # Or just continue until time limit.
            if T < 1e-5:
                break
                
    return best_x
# EVOLVE-BLOCK-END
