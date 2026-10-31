import numpy as np
import time

N = 600
DEADLINE = 120  # seconds

def construct():
    start_time = time.time()
    
    # Initial candidate: Uniform
    # We work with sum 1 internally for optimization efficiency and simplicity.
    best_h = np.ones(N) / N
    
    # Function to compute the "normalized maximum autoconvolution"
    def calc_metric(h):
        # Normalize h to sum to 1
        h_sum = np.sum(h)
        if h_sum == 0:
            return float('inf')
        if abs(h_sum - 1.0) > 1e-9:
            h_norm = h / h_sum
        else:
            h_norm = h
        
        # Use FFT for speed
        # Zero pad to next power of 2 >= 2N-1
        f_len = 2048
        f = np.fft.rfft(h_norm, n=f_len)
        c = np.fft.irfft(f * np.conjugate(f), n=f_len)
        # The autoconvolution is in c. Max is the metric.
        return np.max(c)
    
    best_val = calc_metric(best_h)
    
    np.random.seed(42)
    
    current_h = best_h.copy()
    current_val = best_val
    
    # Adaptive step size for random walk
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < DEADLINE - 5:
        
        # Perturb
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j: 
            continue
        
        # Determine max possible delta to maintain non-negativity
        max_pos_delta = current_h[j]
        max_neg_delta = current_h[i]
        
        # Sample delta based on local scale
        base_step = step_size * max(current_h[i], current_h[j], 1e-9)
        delta = np.random.uniform(-base_step, base_step)
        
        # Clip delta to ensure non-negativity
        if delta > 0:
            if delta > max_pos_delta:
                delta = max_pos_delta
        else:
            if -delta > max_neg_delta:
                delta = -max_neg_delta
                
        if abs(delta) < 1e-12:
            continue
            
        new_h = current_h.copy()
        new_h[i] += delta
        new_h[j] -= delta
        
        # Check non-negativity (safety)
        if np.min(new_h) < -1e-12:
            continue
            
        new_val = calc_metric(new_h)
        
        if new_val < current_val:
            current_h = new_h
            current_val = new_val
            no_improve_count = 0
            if current_val < best_val:
                best_h = current_h.copy()
                best_val = current_val
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Increase step size to escape local minimum
                step_size *= 1.05
                no_improve_count = 0
                if step_size > 0.5:
                    step_size = 0.5

    # Scale result to sum 600 to match seed magnitude, in case evaluator is not scale invariant
    if np.sum(best_h) > 0:
        best_h = best_h * (N / np.sum(best_h))
        
    return best_h.tolist()
