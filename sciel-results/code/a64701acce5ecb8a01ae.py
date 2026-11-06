import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    N_COEFFS = 24
    sqrt_N = math.sqrt(N_COEFFS)
    
    # --- Evaluation Machinery ---
    
    # Precompute indices for vectorized evaluation
    indices = np.arange(N_COEFFS)
    
    def evaluate_fast(coeffs):
        """
        Fast evaluation of max |P(z)| / sqrt(N).
        Uses FFT to find candidate peaks, then GS to refine.
        """
        c_padded = np.zeros(4096, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        # Get top 5 indices
        # We need to be careful with wrap-around, but top 5 is usually safe.
        top_indices = np.argsort(mags)[-5:]
        
        max_val = 0.0
        gr = (math.sqrt(5) - 1) / 2
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / 4096
            # Interval for GS
            a = theta_center - np.pi / 4096
            b = theta_center + np.pi / 4096
            
            # Golden Section Search
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Vectorized evaluation
            # coeffs is a list, convert to array once? 
            # It's better to keep coeffs as np.array for speed in the loop
            c_arr = np.array(coeffs)
            
            def eval_at(theta):
                # z^k = exp(i * k * theta)
                # P(z) = sum c_k exp(i k theta)
                # Use dot product
                return np.abs(np.dot(c_arr, np.exp(1j * indices * theta)))
            
            fc = eval_at(c)
            fd = eval_at(d)
            
            for _ in range(20):
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

    def evaluate_precise(coeffs):
        """
        Higher precision evaluation for final verification or if needed.
        Uses larger FFT and more GS iterations.
        """
        N_FFT = 8192
        c_padded = np.zeros(N_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)
        
        top_indices = np.argsort(mags)[-10:]
        
        max_val = 0.0
        gr = (math.sqrt(5) - 1) / 2
        c_arr = np.array(coeffs)
        
        for idx in top_indices:
            theta_center = 2 * np.pi * idx / N_FFT
            a = theta_center - np.pi / N_FFT
            b = theta_center + np.pi / N_FFT
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def eval_at(theta):
                return np.abs(np.dot(c_arr, np.exp(1j * indices * theta)))
            
            fc = eval_at(c)
            fd = eval_at(d)
            
            for _ in range(30):
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
    
    rs_poly = get_rs_poly(N_COEFFS)
    
    # Generate random sequences
    candidates = [rs_poly]
    for _ in range(20):
        candidates.append([random.choice([1, -1]) for _ in range(N_COEFFS)])
    
    # Evaluate all candidates and pick the best
    best_coeffs = None
    best_score = float('inf')
    
    for cand in candidates:
        score = evaluate_fast(cand)
        if score < best_score:
            best_score = score
            best_coeffs = cand[:]
            
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # --- Iterated Local Search ---
    
    max_kicks = 500 # Safety cap
    kick_count = 0
    
    while time.time() - start_time < deadline - 5:
        # Local Search: Steepest Descent
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
        
        # Perturbation: Random Kick
        if time.time() - start_time > deadline - 5:
            break
            
        # Flip 2-4 random bits
        kick_size = random.randint(2, 4)
        kick_indices = random.sample(range(N_COEFFS), kick_size)
        for k in kick_indices:
            current_coeffs[k] = -current_coeffs[k]
            
        current_score = evaluate_fast(current_coeffs)
        kick_count += 1
        
        # Optional: If we've done many kicks without improving best, try a larger kick or restart
        if kick_count % 50 == 0:
            # Re-evaluate best to ensure it's still good? No, best is fixed.
            # Maybe try a random restart if stuck for long?
            pass
            
    # Final Polish: Run a few more precise checks on the best solution?
    # The evaluator is precise. We just return the best found.
    # However, it's possible that `evaluate_fast` had a slight error in peak location 
    # that led us to a sub-optimal neighbor. 
    # Let's do a final high-precision hill climb from the best_coeffs.
    
    candidate = best_coeffs[:]
    for _ in range(10): # Few iterations of precise hill climb
        improved = False
        for i in range(N_COEFFS):
            new_coeffs = candidate[:]
            new_coeffs[i] = -new_coeffs[i]
            score = evaluate_precise(new_coeffs)
            if score < best_score - 1e-12:
                candidate = new_coeffs
                best_score = score
                best_coeffs = candidate[:]
                improved = True
        if not improved:
            break

    return best_coeffs
