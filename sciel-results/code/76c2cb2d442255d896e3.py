import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    N_COEFFS = 24
    sqrt_N = math.sqrt(N_COEFFS)
    indices = np.arange(N_COEFFS)
    
    def evaluate_fast(coeffs):
        """
        Fast evaluation using FFT and GS.
        """
        M_FFT = 16384
        c_padded = np.zeros(M_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        # Get top 10 indices
        top_indices = np.argsort(mags)[-10:]
        
        max_val = 0.0
        gr = (math.sqrt(5) - 1) / 2
        c_arr = np.array(coeffs)
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / M_FFT
            a = theta_center - np.pi / M_FFT
            b = theta_center + np.pi / M_FFT
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def eval_at(theta):
                return np.abs(np.dot(c_arr, np.exp(1j * indices * theta)))
            
            fc = eval_at(c)
            fd = eval_at(d)
            
            # 25 iterations should be sufficient for high precision
            for _ in range(25):
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
    
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    rs_poly = get_rs_poly(N_COEFFS)
    
    # Try a few random starts and RS
    best_coeffs = None
    best_score = float('inf')
    
    candidates = [rs_poly]
    for _ in range(10):
        candidates.append([random.choice([1, -1]) for _ in range(N_COEFFS)])
        
    for cand in candidates:
        score = evaluate_fast(cand)
        if score < best_score:
            best_score = score
            best_coeffs = cand[:]
            
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # --- Iterated Local Search with 1-flip and 2-flip ---
    
    while time.time() - start_time < deadline - 5:
        # Phase 1: 1-flip local search
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            best_move_score = current_score
            best_move_idx = -1
            
            for i in range(N_COEFFS):
                new_coeffs = current_coeffs[:]
                new_coeffs[i] = -new_coeffs[i]
                score = evaluate_fast(new_coeffs)
                
                if score < best_move_score:
                    best_move_score = score
                    best_move_idx = i
                    improved = True
            
            if best_move_idx != -1:
                current_coeffs[best_move_idx] = -current_coeffs[best_move_idx]
                current_score = best_move_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_coeffs[:]
        
        # Phase 2: 2-flip local search (if time permits)
        if time.time() - start_time < deadline - 10:
            improved = True
            while improved and time.time() - start_time < deadline - 10:
                improved = False
                best_move_score = current_score
                best_move_pair = None
                
                # Try all pairs (i, j) with i < j
                for i in range(N_COEFFS):
                    for j in range(i + 1, N_COEFFS):
                        new_coeffs = current_coeffs[:]
                        new_coeffs[i] = -new_coeffs[i]
                        new_coeffs[j] = -new_coeffs[j]
                        score = evaluate_fast(new_coeffs)
                        
                        if score < best_move_score:
                            best_move_score = score
                            best_move_pair = (i, j)
                            improved = True
                
                if best_move_pair:
                    i, j = best_move_pair
                    current_coeffs[i] = -current_coeffs[i]
                    current_coeffs[j] = -current_coeffs[j]
                    current_score = best_move_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_coeffs = current_coeffs[:]
        
        # Perturbation: Random Kick
        if time.time() - start_time > deadline - 5:
            break
            
        kick_size = random.randint(2, 4)
        kick_indices = random.sample(range(N_COEFFS), kick_size)
        for k in kick_indices:
            current_coeffs[k] = -current_coeffs[k]
            
        current_score = evaluate_fast(current_coeffs)
        
        # If we got worse, we might want to restart from best? 
        # The current loop continues from the kicked state.
        # This is standard ILS.

    return best_coeffs
