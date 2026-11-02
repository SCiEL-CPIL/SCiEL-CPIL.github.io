"""Rewritten program: Simulated Annealing for Erdos minimum overlap."""
import numpy as np
import time

N = 100
DEADLINE = 140  # seconds, safety margin before 150s kill


def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        ov = np.dot(shifted, one_minus_h)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def construct():
    t0 = time.time()
    
    # Initialize with a non-uniform pattern to escape the uniform local minimum
    # Try a few different initializations and keep the best one
    best_h = None
    best_val = float('inf')
    
    inits = [
        np.full(N, 0.5),
        np.sin(np.arange(N) * 2 * np.pi / 10) * 0.4 + 0.5,
        np.sin(np.arange(N) * 2 * np.pi / 5) * 0.4 + 0.5,
        np.random.uniform(0, 1, N)
    ]
    
    for h0 in inits:
        # Project to valid range and fix sum
        h0 = np.clip(h0, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h0)
        if abs(diff) > 1e-10:
            h0 += diff / N
            h0 = np.clip(h0, 0.0, 1.0)
            diff2 = N / 2.0 - np.sum(h0)
            h0 += diff2 / N
        
        val = _max_overlap(h0)
        if val < best_val:
            best_val = val
            best_h = h0.copy()
            
        if time.time() - t0 > DEADLINE - 20:
            break
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_val = _max_overlap(best_h)

    # Simulated Annealing
    h = best_h.copy()
    current_val = best_val
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.99
    
    # Number of steps per temperature
    steps_per_T = 500
    
    while time.time() - t0 < DEADLINE - 5:
        for _ in range(steps_per_T):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Pick random i, j
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
                
            # Random delta
            delta = np.random.uniform(-0.1, 0.1)
            
            h_new = h.copy()
            h_new[i] += delta
            h_new[j] -= delta
            
            # Check bounds
            if h_new[i] < 0.0 or h_new[i] > 1.0 or h_new[j] < 0.0 or h_new[j] > 1.0:
                continue
            
            new_val = _max_overlap(h_new)
            
            # Acceptance probability
            if new_val < current_val:
                accept = True
            else:
                delta_E = new_val - current_val
                if T > 0:
                    prob = np.exp(-delta_E / T)
                    accept = np.random.random() < prob
                else:
                    accept = False
            
            if accept:
                h = h_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                    
        T *= cooling
        if T < T_min:
            break

    # Ensure exact sum for the best found solution
    final_h = best_h.copy()
    final_h = np.clip(final_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(final_h)
    if abs(diff) > 1e-12:
        # Distribute difference
        for i in range(N):
            if diff > 0 and final_h[i] < 1.0:
                add = min(diff, 1.0 - final_h[i])
                final_h[i] += add
                diff -= add
            elif diff < 0 and final_h[i] > 0.0:
                sub = min(-diff, final_h[i])
                final_h[i] -= sub
                diff += sub
            if abs(diff) < 1e-12:
                break
    
    return final_h
