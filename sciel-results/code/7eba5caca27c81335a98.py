import numpy as np
import time
import random

N = 100
# Using a finer quantization to allow better approximation of continuous optimum
# while keeping the search space discrete for SA.
# 51 levels: 0, 1/50, 2/50, ..., 1.0
LEVELS = 51
DEADLINE = 145.0  # Safety margin against 150s limit

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    # h is array of length N
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    # = N/2 - autocorr[k]
    # Compute all autocorrelations via FFT
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    start_time = time.time()
    
    # Initial state: all 0.5 (index 25 in 0..50)
    # Sum must be N/2 = 50.
    # If all are 0.5, sum is 50. This is a valid start.
    h_idx = np.full(N, 25, dtype=int)
    
    def get_h(idx):
        return idx / 50.0

    # Initial evaluation
    h_curr = get_h(h_idx)
    curr_cost = _overlap_max(h_curr)
    
    best_h_idx = h_idx.copy()
    best_cost = curr_cost
    
    # Simulated Annealing parameters
    # We need to maintain sum(h) = N/2.
    # Strategy: Pick two random indices i, j.
    # Change h[i] by delta, h[j] by -delta to keep sum constant.
    # Ensure values stay in [0, 1].
    
    T = 0.5
    T_min = 1e-5
    cooling_rate = 0.999
    
    # Precompute possible moves? No, just random.
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        # Pick two distinct indices
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
            
        # Pick a step size in index space (1 unit = 0.02 in height)
        # Try small steps first, occasionally larger?
        # Let's stick to step = 1 for stability, or maybe 2.
        step = 1
        
        # Determine direction: increase i, decrease j OR decrease i, increase j
        if random.random() < 0.5:
            new_i = h_idx[i] + step
            new_j = h_idx[j] - step
        else:
            new_i = h_idx[i] - step
            new_j = h_idx[j] + step
            
        # Check bounds
        if new_i < 0 or new_i >= LEVELS or new_j < 0 or new_j >= LEVELS:
            continue
            
        # Apply move
        old_i = h_idx[i]
        old_j = h_idx[j]
        
        h_idx[i] = new_i
        h_idx[j] = new_j
        
        # Evaluate new cost
        # Optimization: We can compute the change in overlap more efficiently?
        # No, FFT is fast enough for N=100.
        h_new = get_h(h_idx)
        new_cost = _overlap_max(h_new)
        
        delta = new_cost - curr_cost
        
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            curr_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h_idx = h_idx.copy()
        else:
            # Revert
            h_idx[i] = old_i
            h_idx[j] = old_j
            
        T *= cooling_rate
        
    # Return the best found heights
    return get_h(best_h_idx)
