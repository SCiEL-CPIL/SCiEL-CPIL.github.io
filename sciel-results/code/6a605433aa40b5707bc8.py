import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def compute_value(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    # Initial state: Use a structure that is not uniform but has some variation
    # Start with a "triangle wave" or similar to break symmetry, then let SA find the best
    # Actually, starting from ones is safe but might be a local trap.
    # Let's start with a random signal normalized to sum 1.
    np.random.seed(42)
    h = np.random.randn(N)
    # Ensure sum is not zero
    if np.sum(h) == 0:
        h[0] += 1e-9
    # Normalize to sum 1 to keep scale consistent initially? 
    # The metric is scale invariant: 
    # value = |2N * max(conv(h,h)) / sum(h)^2|
    # If h -> c*h, conv -> c^2*conv, sum -> c*sum, sum^2 -> c^2*sum^2.
    # So value is invariant to scaling. 
    # However, numerical stability might vary. Let's keep sum(h) = 1 for the search.
    h = h / np.sum(h)
    
    current_val = compute_value(h)
    best_h = h.copy()
    best_val = current_val
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    
    # We will use block perturbations. 
    # A single element change might be too small. A block of size 10-50 is better.
    
    steps = 0
    T = T_start
    
    while time.time() < deadline:
        steps += 1
        
        # Generate a perturbation
        # Randomly choose a block size and location
        block_size = np.random.randint(5, 50)
        start_idx = np.random.randint(0, N - block_size + 1)
        
        # Create a perturbation vector
        # We can add noise or shift the block.
        # Let's try adding Gaussian noise to a block, then re-normalizing the whole signal
        # to keep sum=1 (optional, but keeps search in a compact space).
        
        h_new = h.copy()
        noise = np.random.randn(block_size) * 0.1
        h_new[start_idx:start_idx + block_size] += noise
        
        # Re-normalize to sum=1 to prevent drift and keep scale invariant
        s_new = np.sum(h_new)
        if s_new == 0 or not np.isfinite(s_new):
            continue
        h_new = h_new / s_new
        
        new_val = compute_value(h_new)
        
        # Acceptance criterion
        if new_val < current_val:
            h = h_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            # Metropolis criterion
            # delta = new_val - current_val
            # Accept with probability exp(-delta / T)
            if T > 0:
                delta = new_val - current_val
                # If delta is very large, exp(-delta/T) underflows to 0
                if delta < 100 * T: 
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        h = h_new
                        current_val = new_val
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Check time more frequently in the inner loop? 
        # The loop runs fast. 115s is plenty for many iterations.
        # Let's add a hard break if we exceed deadline
        if time.time() > deadline - 5:
            break

    # Final polish: try a few random restarts from best_h if time permits?
    # Or just return best_h.
    # Given the strict time limit and the nature of SA, returning the best found is standard.
    
    # Ensure the output is a sequence of 400 signed step heights
    # The problem asks for "real, possibly negative step heights".
    # Our h is normalized to sum 1. This is a valid set of heights.
    
    return best_h
