import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    N_COEFFS = 24
    SQRT_N = math.sqrt(N_COEFFS)
    N_FFT = 4096
    
    # Precompute FFT setup
    k = np.arange(N_COEFFS, dtype=np.float64)
    
    def fast_score(coeffs):
        """Fast approximate score using FFT only (no refinement)."""
        c = np.zeros(N_FFT, dtype=np.complex128)
        c[:N_COEFFS] = coeffs
        vals = np.fft.fft(c)
        return np.max(np.abs(vals))
    
    def precise_score(coeffs):
        """Precise score with golden-section refinement."""
        c = np.zeros(N_FFT, dtype=np.complex128)
        c[:N_COEFFS] = coeffs
        vals = np.fft.fft(c)
        mags = np.abs(vals)
        
        # Find top 5 peaks
        top_indices = np.argsort(mags)[::-1][:5]
        
        coeffs_arr = np.array(coeffs, dtype=np.float64)
        
        def refine_peak(a, b):
            gr = (math.sqrt(5) - 1) / 2
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k * c_pt)))
            fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k * d_pt)))
            for _ in range(40):
                if fc < fd:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k * d_pt)))
                else:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k * c_pt)))
            return max(fc, fd)
        
        max_val = 0.0
        for idx in top_indices:
            theta_max = 2 * np.pi * idx / N_FFT
            a = theta_max - np.pi / N_FFT
            b = theta_max + np.pi / N_FFT
            refined = refine_peak(a, b)
            if refined > max_val:
                max_val = refined
        
        return max_val / SQRT_N
    
    # Generate initial seeds
    def get_rudin_shapiro(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    seeds = []
    # Rudin-Shapiro
    seeds.append(get_rudin_shapiro(N_COEFFS))
    
    # Random seeds
    random.seed(42)
    for _ in range(20):
        seeds.append([random.choice([1, -1]) for _ in range(N_COEFFS)])
    
    # Some structured seeds
    # Alternating blocks
    seeds.append([1 if (i // 4) % 2 == 0 else -1 for i in range(N_COEFFS)])
    seeds.append([1 if (i // 6) % 2 == 0 else -1 for i in range(N_COEFFS)])
    seeds.append([1 if (i // 3) % 2 == 0 else -1 for i in range(N_COEFFS)])
    
    # Evaluate all seeds with fast score
    best_coeffs = None
    best_fast = float('inf')
    
    for seed in seeds:
        s = fast_score(seed)
        if s < best_fast:
            best_fast = s
            best_coeffs = seed[:]
    
    current = best_coeffs[:]
    current_fast = best_fast
    
    # Steepest descent with multiple restarts
    def local_search(coeffs, max_iter=500):
        """Steepest descent: try all single-bit flips, keep best."""
        current = coeffs[:]
        current_score = fast_score(current)
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            best_flip = -1
            best_new_score = current_score
            
            for i in range(N_COEFFS):
                # Try flipping bit i
                new_coeffs = current[:]
                new_coeffs[i] = -new_coeffs[i]
                s = fast_score(new_coeffs)
                if s < best_new_score:
                    best_new_score = s
                    best_flip = i
            
            if best_flip == -1:
                break  # Local minimum
            
            current[best_flip] = -current[best_flip]
            current_score = best_new_score
        
        return current, current_score
    
    # Run local search from best seed
    result, result_score = local_search(current)
    best_coeffs = result[:]
    best_fast = result_score
    
    # Try more restarts
    random.seed(123)
    for restart in range(50):
        if time.time() - start_time > deadline:
            break
        
        # Random restart
        seed = [random.choice([1, -1]) for _ in range(N_COEFFS)]
        
        # Quick local search
        result, result_score = local_search(seed, max_iter=200)
        
        if result_score < best_fast:
            best_fast = result_score
            best_coeffs = result[:]
    
    # Try perturbations of the best found
    for _ in range(100):
        if time.time() - start_time > deadline:
            break
        
        # Perturb best by flipping 2-3 random bits
        perturbed = best_coeffs[:]
        for _ in range(random.randint(2, 3)):
            i = random.randint(0, N_COEFFS - 1)
            perturbed[i] = -perturbed[i]
        
        result, result_score = local_search(perturbed, max_iter=300)
        
        if result_score < best_fast:
            best_fast = result_score
            best_coeffs = result[:]
    
    # Final: do precise evaluation on best candidate
    # Also try a few neighbors of the best
    candidates = [best_coeffs]
    for i in range(N_COEFFS):
        neighbor = best_coeffs[:]
        neighbor[i] = -neighbor[i]
        candidates.append(neighbor)
    
    best_precise = float('inf')
    best_final = best_coeffs
    
    for cand in candidates:
        if time.time() - start_time > deadline:
            break
        ps = precise_score(cand)
        if ps < best_precise:
            best_precise = ps
            best_final = cand
    
    return best_final
