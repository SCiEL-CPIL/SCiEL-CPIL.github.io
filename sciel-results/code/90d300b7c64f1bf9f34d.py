import numpy as np
import time
import random

def construct():
    """
    Returns the 24 coefficients of the polynomial P(z) = sum c_k z^k 
    that minimizes the sup norm on the unit circle.
    """
    # Precompute theta grid for FFT-based evaluation
    N = 1 << 14  # 16384 points, power of 2 for fast FFT
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    def fft_eval(coeffs):
        """Fast evaluation using FFT. Returns |P(z)| at N points on unit circle."""
        c = np.array(coeffs, dtype=np.float64)
        # Pad to length N
        c_padded = np.zeros(N, dtype=np.float64)
        c_padded[:24] = c
        # FFT gives values at z^k = exp(2πi k/N) for k=0..N-1
        # But we need P(z_n) = sum c_k z_n^k where z_n = exp(2πi n/N)
        # This is the DFT of c_padded
        vals = np.fft.fft(c_padded)
        return np.abs(vals)
    
    def find_peak(coeffs):
        """Find the maximum |P(z)| on unit circle with golden-section refinement."""
        mags = fft_eval(coeffs)
        idx = np.argmax(mags)
        step = 2 * np.pi / N
        t_low = thetas[(idx - 1) % N]
        t_high = thetas[(idx + 1) % N]
        
        c_arr = np.array(coeffs, dtype=np.float64)
        
        def neg_sq_mag(t):
            zt = np.exp(1j * t)
            # Compute P(zt) = sum c_k zt^k
            zt_powers = np.array([zt**k for k in range(24)])
            pt = np.dot(c_arr, zt_powers)
            return -np.abs(pt)**2
            
        gr = (np.sqrt(5) - 1) / 2
        a, b = t_low, t_high
        c_ = b - gr * (b - a)
        d_ = a + gr * (b - a)
        fc = neg_sq_mag(c_)
        fd = neg_sq_mag(d_)
        
        for _ in range(50):
            if fc > fd:
                b = d_
                d_ = c_
                fd = fc
                c_ = b - gr * (b - a)
                fc = neg_sq_mag(c_)
            else:
                a = c_
                c_ = d_
                fc = fd
                d_ = a + gr * (b - a)
                fd = neg_sq_mag(d_)
                
        t_opt = (a + b) / 2
        z_opt = np.exp(1j * t_opt)
        z_opt_powers = np.array([z_opt**k for k in range(24)])
        p_opt = np.dot(c_arr, z_opt_powers)
        return np.abs(p_opt) / np.sqrt(24)
    
    def eval_fast(coeffs):
        """Fast evaluation using FFT (coarse, no refinement)."""
        mags = fft_eval(coeffs)
        return np.max(mags) / np.sqrt(24)
    
    def eval_refine(coeffs):
        """Evaluation with golden-section refinement."""
        return find_peak(coeffs)
    
    # Known good starting points (from Odlyzko's tables and literature)
    candidates = [
        [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1],
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1],
        [1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1],
    ]
    
    OPT_VAL = 1.2583758
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Local search using FFT-based evaluation
    def local_search(coeffs, time_limit=5.0):
        current_coeffs = list(coeffs)
        current_val = eval_fast(current_coeffs)
        improved = True
        start_time_ls = time.time()
        
        while improved and (time.time() - start_time_ls) < time_limit:
            improved = False
            indices = list(range(24))
            random.shuffle(indices)
            for k in indices:
                new_coeffs = current_coeffs[:]
                new_coeffs[k] *= -1
                new_val = eval_fast(new_coeffs)
                if new_val < current_val - 1e-12:
                    current_coeffs = new_coeffs
                    current_val = new_val
                    improved = True
                    
        return current_coeffs, current_val
    
    # Evaluate initial candidates
    best_coeffs = None
    best_val = float('inf')
    
    for cand in candidates:
        val = eval_fast(cand)
        if val < best_val:
            best_val = val
            best_coeffs = list(cand)
    
    # Local search from best candidate
    best_coeffs, best_val = local_search(best_coeffs, time_limit=10.0)
    
    # If we are already at the optimum (within tolerance), return
    if best_val <= OPT_VAL + 1e-6:
        final_val = eval_refine(best_coeffs)
        return best_coeffs
    
    # Random restarts with local search
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Generate random coefficients
        rand_coeffs = [random.choice([1, -1]) for _ in range(24)]
        
        # Local search
        c, v = local_search(rand_coeffs, time_limit=3.0)
        
        if v < best_val - 1e-12:
            best_coeffs = c
            best_val = v
            
            # Check if we hit the global optimum
            if best_val <= OPT_VAL + 1e-6:
                break
                
    # Final high-precision polish: try flipping pairs of coefficients
    if time.time() < deadline - 5:
        polished = list(best_coeffs)
        polished_val = eval_fast(polished)
        improved = True
        polish_deadline = min(time.time() + 20, deadline)
        
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(24):
                for j in range(i+1, 24):
                    new_coeffs = polished[:]
                    new_coeffs[i] *= -1
                    new_coeffs[j] *= -1
                    new_val = eval_fast(new_coeffs)
                    if new_val < polished_val - 1e-12:
                        polished = new_coeffs
                        polished_val = new_val
                        improved = True
                        if polished_val <= OPT_VAL + 1e-6:
                            break
                if polished_val <= OPT_VAL + 1e-6:
                    break
        
        if polished_val < best_val:
            best_coeffs = polished
            best_val = polished_val
    
    return best_coeffs
