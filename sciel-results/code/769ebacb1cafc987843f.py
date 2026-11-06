import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    n_coeffs = 24
    sqrt_n = np.sqrt(n_coeffs)
    
    # Fast scoring with smaller FFT for inner loop
    fft_size_fast = 1024
    fft_size_precise = 4096
    
    def score_fast(coeffs):
        f = np.fft.fft(coeffs, n=fft_size_fast)
        return np.max(np.abs(f)) / sqrt_n
    
    def score_precise(coeffs):
        f = np.fft.fft(coeffs, n=fft_size_precise)
        return np.max(np.abs(f)) / sqrt_n
    
    def hill_climb_enhanced(initial_coeffs):
        current = initial_coeffs.copy()
        current_score = score_fast(current)
        improved = True
        
        # Phase 1: Single flips
        while improved and (time.time() - start_time < time_limit - 10):
            improved = False
            indices = list(range(n_coeffs))
            random.shuffle(indices)
            
            for i in indices:
                old_val = current[i]
                current[i] = -old_val
                new_score = score_fast(current)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_val
        
        # Phase 2: Pairwise swaps (flip two at once)
        for _ in range(3):
            if time.time() - start_time > time_limit - 10:
                break
            improved = True
            while improved:
                improved = False
                # Try all pairs
                for i in range(n_coeffs - 1):
                    for j in range(i + 1, n_coeffs):
                        old_i, old_j = current[i], current[j]
                        current[i] = -old_i
                        current[j] = -old_j
                        new_score = score_fast(current)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            current[i] = old_i
                            current[j] = old_j
        
        return current, score_precise(current)
    
    best_coeffs = None
    best_score = float('inf')
    
    # Generate diverse initial candidates
    candidates = []
    
    # All ones
    candidates.append([1] * n_coeffs)
    
    # Golay-Rudin-Shapiro
    gr_p, gr_q = [1], [1]
    while len(gr_p) < n_coeffs:
        gr_p, gr_q = gr_p + gr_q, gr_p + [-c for c in gr_q]
    candidates.append(gr_p[:n_coeffs])
    
    # Some structured patterns
    # Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n_coeffs)])
    
    # More structured patterns based on known good sequences
    # Try patterns with balanced +1 and -1
    for pattern_seed in range(8):
        random.seed(pattern_seed)
        cand = [random.choice([1, -1]) for _ in range(n_coeffs)]
        candidates.append(cand)
    
    # Random candidates
    for _ in range(15):
        candidates.append([random.choice([1, -1]) for _ in range(n_coeffs)])
    
    # Run enhanced hill climbing on all candidates
    for cand in candidates:
        if time.time() - start_time > time_limit - 5:
            break
        coeffs, sc = hill_climb_enhanced(cand)
        if sc < best_score:
            best_score = sc
            best_coeffs = coeffs.copy()
    
    # More random restarts with enhanced hill climbing
    while time.time() - start_time < time_limit - 5:
        random_seed = [random.choice([1, -1]) for _ in range(n_coeffs)]
        coeffs, sc = hill_climb_enhanced(random_seed)
        if sc < best_score:
            best_score = sc
            best_coeffs = coeffs.copy()
    
    if best_coeffs is None:
        return gr_p[:n_coeffs]
    
    return best_coeffs
