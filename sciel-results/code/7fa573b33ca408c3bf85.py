"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_c3(x):
    """
    Compute the upper bound on C3 for a given step function x (length N).
    The autoconvolution is (x * x)[k] = sum_i x[i] x[i+k].
    The bound is max_k |(x*x)[k]|.
    """
    # Use FFT for efficient autoconvolution
    n_fft = 1 << (2 * N - 1).bit_length()
    xf = np.fft.rfft(x, n=n_fft)
    conv = np.fft.irfft(xf * xf, n=n_fft)[:N]
    # The autoconvolution in the circular sense vs linear:
    # For step functions on [-1/4, 1/4] with N intervals, the "autoconvolution" 
    # relevant to the C3 bound typically involves the sum over shifts.
    # Let's assume the standard definition: C(x) = max_k |sum_i x_i x_{i+k}|
    # where indices are modulo N (circular) or linear?
    # Given "step function on [-1/4, 1/4]", it's a periodic or compact support?
    # "400 equal intervals on [-1/4, 1/4]" suggests compact support, so linear convolution.
    # However, for signed step functions in such contexts, often the circular autocorrelation
    # or the specific sum defined in the "third autocorrelation inequality" is used.
    # Let's look at the score: "upper bound on C3".
    # Common C3 bound involves max over shifts of |sum x_i x_{i+k}|.
    # Let's compute the linear autoconvolution max absolute value.
    # But wait, the FFT above computes circular convolution if we don't zero-pad properly.
    # We zero-padded to n_fft, so `conv` contains the linear autoconvolution in the first 2N-1 elements.
    # The relevant shifts are usually 0 to N-1 (or -N+1 to N-1).
    # Let's take the max abs over the first N elements (shifts 0..N-1).
    return np.max(np.abs(conv[:N]))

def _objective(x):
    return _compute_c3(x)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: random signs
    np.random.seed(42)
    x0 = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x0.copy()
    best_score = _compute_c3(best_x)
    
    # Strategy:
    # 1. Local search with multi-flip moves (single and double flips)
    # 2. Vectorized evaluation where possible
    
    # Precompute indices for efficiency?
    # _compute_c3 is fast with FFT.
    
    # Local Search
    # We will try single flips and double flips.
    # To speed up, we can try to estimate the change, but FFT is O(N log N) which is fast enough for N=400.
    # N=400, FFT size ~1024. Very fast.
    
    # Let's do a simple hill climbing with restarts if stuck.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x = x0.copy()
        else:
            # Random restart
            x = np.random.choice([-1.0, 1.0], size=N)
            
        current_score = _compute_c3(x)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try single flips
            # We can vectorize the flip attempts?
            # Not easily with the FFT objective.
            # But we can try a subset of flips per iteration to save time if N is large.
            # For N=400, trying all 400 flips is 400 FFTs.
            # Each FFT is ~10us? 400 * 10us = 4ms.
            # So we can do many iterations.
            
            best_flip_idx = -1
            best_flip_score = current_score
            
            # Try all single flips
            for i in range(N):
                x[i] = -x[i]
                s = _compute_c3(x)
                x[i] = -x[i] # revert
                
                if s < best_flip_score:
                    best_flip_score = s
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                x[best_flip_idx] = -x[best_flip_idx]
                current_score = best_flip_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            
            # If no single flip improves, try double flips (optional, might be too slow)
            # Or break.
            # Let's add a small random perturbation if stuck for a while?
            # For now, break out of inner loop if no single flip works.
            # This is standard hill climbing.
        
        # If we improved global best, we might want to continue from here?
        # The loop structure restarts from random if restart > 0.
        # If restart == 0, we did one full hill climb.
        
        # To be more efficient, we can check if we are close to the deadline and stop.
        
    return best_x

# EVOLVE-BLOCK-END
