"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    h = 0.5 / N
    centers = -0.25 + (np.arange(N) + 0.5) * h
    
    def compute_score(heights):
        n_fft = 2048
        F = np.fft.rfft(heights, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * h
        max_conv = np.max(conv)
        total_mass = np.sum(heights) * h
        return max_conv / (total_mass ** 2)
    
    # Try various parametric families to find a good initial guess
    best_val = float('inf')
    best_heights = None
    
    # Family 1: Gaussians
    sigmas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
    for sigma in sigmas:
        heights = np.exp(-centers**2 / (2 * sigma**2))
        val = compute_score(heights)
        if val < best_val:
            best_val = val
            best_heights = heights.copy()
        
        if time.time() > deadline - 10:
            break
            
    # Family 2: Super-Gaussians (1 - (x/0.25)^2)^p
    ps = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for p in ps:
        base = 1 - (centers / 0.25)**2
        base = np.maximum(base, 0)
        heights = base ** p
        val = compute_score(heights)
        if val < best_val:
            best_val = val
            best_heights = heights.copy()
            
        if time.time() > deadline - 10:
            break
            
    # Family 3: Cosine
    for k in [1, 2, 3]:
        heights = np.cos(k * np.pi * centers / 0.25)
        heights = np.maximum(heights, 0)
        val = compute_score(heights)
        if val < best_val:
            best_val = val
            best_heights = heights.copy()

    if best_heights is None:
        return [1.0] * N

    # Local optimization with incremental autoconvolution update
    current = best_heights.copy()
    current_val = best_val
    
    # Precompute FFT size
    n_fft = 2048
    
    # Compute initial autoconvolution
    F = np.fft.rfft(current, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * h
    max_conv = np.max(conv)
    total_mass = np.sum(current) * h
    current_score = max_conv / (total_mass ** 2)
    
    # Incremental update logic
    # (f + delta * e_i) * (f + delta * e_i) = f*f + delta * (e_i * f + f * e_i) + delta^2 * (e_i * e_i)
    # e_i is a delta function at index i.
    # e_i * f is f shifted by i.
    # In frequency domain, this is easier, but spatial domain update is O(N).
    # Let's stick to FFT for simplicity unless we need extreme speed.
    # Given N=600, FFT is ~10us. 10000 evals = 0.1s.
    # We can afford many FFTs.
    
    np.random.seed(42)
    step_size = 0.01
    
    # Coordinate descent style
    for _ in range(500): # Max passes
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Randomize order to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            # Try increasing
            if current[i] < 10.0: # Cap height
                old_h = current[i]
                current[i] = old_h + step_size
                
                F = np.fft.rfft(current, n=n_fft)
                conv = np.fft.irfft(F * F, n=n_fft) * h
                max_conv = np.max(conv)
                total_mass = np.sum(current) * h
                new_score = max_conv / (total_mass ** 2)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_val:
                        best_val = current_score
                        best_heights = current.copy()
                else:
                    current[i] = old_h
            
            # Try decreasing
            if current[i] > step_size:
                old_h = current[i]
                current[i] = old_h - step_size
                
                F = np.fft.rfft(current, n=n_fft)
                conv = np.fft.irfft(F * F, n=n_fft) * h
                max_conv = np.max(conv)
                total_mass = np.sum(current) * h
                new_score = max_conv / (total_mass ** 2)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_val:
                        best_val = current_score
                        best_heights = current.copy()
                else:
                    current[i] = old_h
            
            if not improved and _ > 10:
                # Stagnation, try larger step or break
                step_size *= 1.5
                if step_size > 0.5:
                    break

    if best_heights is None:
        return [1.0] * N
    return list(best_heights)
# EVOLVE-BLOCK-END
