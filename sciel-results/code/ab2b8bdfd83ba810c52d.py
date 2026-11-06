import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    N_COEFFS = 24
    sqrt_N = math.sqrt(N_COEFFS)
    
    # Precompute indices for vectorized evaluation in GS
    indices = np.arange(N_COEFFS)
    
    # FFT size for dense sampling
    M_FFT = 16384
    
    def evaluate_fast(coeffs):
        """
        Fast evaluation of max |P(z)| / sqrt(N).
        Uses FFT to find candidate peaks, then GS to refine.
        """
        c_padded = np.zeros(M_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        # Get top 3 indices to be safe
        # argsort is O(M log M), but M is small. 
        # For speed, we can just find the global max and its neighbors.
        # However, sometimes the true max is between two FFT bins.
        # Top 3 covers this well.
        top_indices = np.argsort(mags)[-3:]
        
        max_val = 0.0
        gr = (math.sqrt(5) - 1) / 2
        
        c_arr = np.array(coeffs)
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / M_FFT
            # Interval for GS
            a = theta_center - np.pi / M_FFT
            b = theta_center + np.pi / M_FFT
            
            # Golden Section Search
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def eval_at(theta):
                return np.abs(np.dot(c_arr, np.exp(1j * indices * theta)))
            
            fc = eval_at(c)
            fd = eval_at(d)
            
            # 15 iterations is usually enough for 1e-9 accuracy on smooth functions
            for _ in range(15):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = eval_at(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = eval_at(c)
            
            local_max = max(fc, fd)
            if local_max > max_val:
                max_val = local_max
                
        return max_val / sqrt_N

    # --- Initialization ---
    
    # Rudin-Shapiro
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    # Start with RS and a few randoms to pick the best initial
    candidates = [get_rs_poly(N_COEFFS)]
    for _ in range(5):
        candidates.append([random.choice([1, -1]) for _ in range(N_COEFFS)])
        
    best_coeffs = None
    best_score = float('inf')
    
    for cand in candidates:
        score = evaluate_fast(cand)
        if score < best_score:
            best_score = score
            best_coeffs = cand[:]
            
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # --- Simulated Annealing ---
    
    # Temperature parameters
    # Scores are around 1.25. Differences between neighbors are small.
    # Start T high enough to accept worse moves, cool down.
    T = 0.1 
    cooling_rate = 0.999
    min_T = 1e-5
    
    # To prevent getting stuck, we track iterations since last improvement
    iter_no_improve = 0
    max_iter_no_improve = 500
    
    while time.time() - start_time < deadline:
        # Flip one random bit
        idx = random.randint(0, N_COEFFS - 1)
        current_coeffs[idx] = -current_coeffs[idx]
        
        new_score = evaluate_fast(current_coeffs)
        
        delta = new_score - current_score
        
        # Acceptance probability
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_coeffs = current_coeffs[:]
                iter_no_improve = 0
            else:
                iter_no_improve += 1
        else:
            # Revert
            current_coeffs[idx] = -current_coeffs[idx]
            iter_no_improve += 1
            
        # Cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Restart logic: if stuck, reset to best and cool down
        if iter_no_improve > max_iter_no_improve:
            current_coeffs = best_coeffs[:]
            current_score = best_score
            T = 0.1 # Reset temperature
            iter_no_improve = 0
            max_iter_no_improve = random.randint(300, 800) # Vary patience

    return best_coeffs
