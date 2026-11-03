import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 600
DEADLINE = 145.0 # Leave 5s margin

def construct():
    start_time = time.time()
    
    def get_score(h):
        if np.all(h == 0):
            return float('inf')
        # Compute autoconvolution using FFT
        # Zero padding to avoid circular convolution effects?
        # Linear convolution of two N-length sequences requires 2N-1 points.
        # FFT size should be at least 2N-1.
        # N=600, 2N-1=1199. Next power of 2 is 2048.
        fft_h = np.fft.rfft(h, n=2048)
        conv = np.fft.irfft(fft_h * fft_h, n=2048)
        # The linear convolution is in the first 2N-1 points
        conv_lin = conv[:2*N-1]
        peak = np.max(conv_lin)
        norm_sq = np.sum(h**2)
        return peak / norm_sq

    def coordinate_descent(h, time_limit):
        h = h.copy()
        current_score = get_score(h)
        
        # Simple coordinate descent
        # We iterate over indices.
        # For each index, we optimize h[i] using minimize_scalar.
        
        while (time.time() - start_time) < time_limit:
            improved = False
            for i in range(N):
                if (time.time() - start_time) > time_limit:
                    break
                
                # Determine bounds for h[i]
                # Use current max as upper bound multiplier
                max_val = np.max(h) * 1.5 + 0.1
                if max_val < 1e-6:
                    max_val = 1.0
                    
                def obj_1d(x):
                    old_val = h[i]
                    h[i] = x
                    s = get_score(h)
                    h[i] = old_val
                    return s
                
                # minimize_scalar might be slow if called N times per epoch
                # Let's use a smaller tolerance
                res = minimize_scalar(obj_1d, bounds=(0, max_val), method='bounded', options={'xatol': 1e-5})
                new_val = res.x
                h[i] = new_val
                new_score = get_score(h)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
            
            if not improved:
                break
        
        return h, current_score

    candidates = []
    
    # 1. Semicircle
    x = np.linspace(-1, 1, N)
    h_semi = np.sqrt(np.maximum(0, 1 - x**2))
    candidates.append(h_semi)
    
    # 2. Triangle
    h_tri = np.maximum(0, 1 - np.abs(x))
    candidates.append(h_tri)
    
    # 3. Rectangle
    h_rect = np.ones(N)
    candidates.append(h_rect)
    
    # 4. Asymmetric Semicircle
    # Shift the center to break symmetry
    x_shift = np.linspace(-1.5, 0.5, N)
    h_asym = np.sqrt(np.maximum(0, 1 - x_shift**2))
    candidates.append(h_asym)
    
    # 5. Random
    np.random.seed(42)
    h_rand = np.random.rand(N)
    candidates.append(h_rand)

    best_h = None
    best_score = float('inf')
    
    # Allocate time for each candidate
    # Total 145s. 5 candidates. ~25s each.
    time_per_candidate = 25.0
    
    for idx, h_init in enumerate(candidates):
        # Check time
        if time.time() > DEADLINE - 5:
            break
            
        # Normalize
        norm = np.linalg.norm(h_init)
        if norm > 0:
            h_init = h_init / norm
        else:
            continue
            
        h_opt, score = coordinate_descent(h_init, start_time + time_per_candidate)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
            
        # If we found a score < 1.0, we might be able to stop early if we have plenty of time?
        # No, let's try to improve further.

    if best_h is None:
        return [1.0] * N
        
    best_h = np.maximum(0, best_h)
    return list(best_h)
