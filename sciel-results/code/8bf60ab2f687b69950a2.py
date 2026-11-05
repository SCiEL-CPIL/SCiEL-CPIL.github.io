import numpy as np
import time
import random

def construct():
    # Start time
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate initial Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=float)
    
    # 2. Define fast sup norm evaluation using FFT
    M = 2**15  # 32768 points, enough resolution for N=24
    def get_supnorm(coeffs):
        # coeffs is a numpy array of +1/-1
        # Pad to M
        c_padded = np.zeros(M, dtype=float)
        c_padded[:24] = coeffs
        F = np.fft.fft(c_padded)
        return float(np.max(np.abs(F)))
    
    # 3. Simulated Annealing
    current_c = c.copy()
    current_val = get_supnorm(current_c)
    best_c = current_c.copy()
    best_val = current_val
    
    # Temperature tuning
    # Typical change in max norm is small. 
    # Let's start T at 0.05 and decay to 0.001
    T_start = 0.05
    T_end = 0.001
    n_steps = 500000  # Rough estimate, will be cut by time
    
    # We don't know exact steps, so we loop until deadline
    step = 0
    while time.time() < deadline:
        # Flip a random bit
        idx = random.randint(0, 23)
        new_c = current_c.copy()
        new_c[idx] *= -1
        
        new_val = get_supnorm(new_c)
        delta = new_val - current_val
        
        # Acceptance probability
        if delta < 0 or (random.random() < np.exp(-delta / T_start)):
            current_c = new_c
            current_val = new_val
            if current_val < best_val:
                best_c = current_c.copy()
                best_val = current_val
        
        step += 1
        # Decay temperature based on step count or time
        # Linear decay over estimated total steps or time
        # Let's use time-based decay
        elapsed = time.time() - start_time
        total_time = 110.0
        progress = elapsed / total_time
        T = T_start * (T_end / T_start) ** progress
        
        # If T is too small, we might get stuck, but for N=24 it should be fine.
        # Also, if we find a very good value, we might want to restart?
        # No, keep it simple.
        
    # Return the best coefficients as integers
    return [int(x) for x in best_c]
