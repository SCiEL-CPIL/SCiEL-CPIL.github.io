import numpy as np
from scipy.optimize import linprog
import time

N = 600
INTERVALS = N
TIME_LIMIT = 140  # Safety margin (evaluator kills at 150s)

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution of a step function.
    The step function has N equal intervals on [-1/4, 1/4].
    The autoconvolution is computed in the discrete sense.
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-12):
        return float('inf')
    
    # Discrete autoconvolution (full)
    # For step functions on equal intervals, the autoconvolution at lag k
    # is the sum of products of adjacent values.
    # Using FFT for efficiency
    n = len(f)
    # Pad to avoid circular convolution
    size = 2 * n - 1
    f_ext = np.zeros(2 * size)
    f_ext[:n] = f
    
    # Autoconvolution via FFT
    F = np.fft.rfft(f_ext, n=2*size)
    conv = np.fft.irfft(F * F, n=2*size)
    # The linear autoconvolution of f with itself (reversed) is at the center
    # Actually, autoconvolution in signal processing usually means f * f (not f * f_flipped)
    # Let's clarify: "autoconvolution" in the context of the first autocorrelation inequality
    # typically refers to the convolution of f with itself, i.e., (f * f)(x) = ∫ f(t) f(x-t) dt
    
    # For discrete step functions on equal intervals, the convolution at each point
    # can be computed. The maximum of the normalized autoconvolution is what we want.
    # Normalization: typically by ||f||_2^2 or similar.
    
    # Let's use the standard definition:
    # autoconv = np.convolve(f, f, mode='full')
    # normalized_max = max(autoconv) / (np.sum(f**2))
    # But we need to be careful about the exact normalization.
    
    # For step functions with unit interval width, the continuous autoconvolution
    # is approximated by the discrete convolution of the heights.
    autoconv = np.convolve(f, f, mode='full')
    l2_norm_sq = np.sum(f**2)
    
    if l2_norm_sq < 1e-15:
        return float('inf')
    
    normalized_max = np.max(autoconv) / l2_norm_sq
    return normalized_max

def construct():
    """
    Use linear programming to find nonnegative step heights that minimize
    the normalized maximum of the autoconvolution.
    """
    start_time = time.time()
    
    # Initial guess: constant function
    f0 = np.ones(N)
    best_f = f0.copy()
    best_score = compute_score(best_f)
    
    # If we're running out of time, return the best we have
    if time.time() - start_time > TIME_LIMIT - 5:
        return best_f.tolist()
    
    # Try to improve using coordinate descent with local search
    # First, let's try a simple local search approach
    for it in range(100):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Random perturbation
        idx = np.random.randint(N)
        delta = 0.1 * best_f[idx] if best_f[idx] > 0 else 0.1
        
        f_new = best_f.copy()
        f_new[idx] = max(0, best_f[idx] + delta)
        
        score_new = compute_score(f_new)
        if score_new < best_score:
            best_f = f_new
            best_score = score_new
    
    # Try more sophisticated approach: minimize the maximum using iterative reweighting
    for it in range(50):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Compute current autoconvolution
        autoconv = np.convolve(best_f, best_f, mode='full')
        l2_norm_sq = np.sum(best_f**2)
        if l2_norm_sq < 1e-15:
            break
        
        normalized_conv = autoconv / l2_norm_sq
        max_val = np.max(normalized_conv)
        
        # Find where the maximum occurs
        max_idx = np.argmax(normalized_conv)
        
        # Try to reduce the peak by adjusting f
        # The gradient of the autoconvolution with respect to f[i] at lag k
        # is f[k-i] (approximately)
        
        # Simple approach: reduce f at positions that contribute to the peak
        # and increase elsewhere
        
        # Use a simple iterative method: scale down the peak-contributing elements
        # and scale up others
        
        # For now, let's try a simpler local search
        improved = False
        for idx in range(N):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Try decreasing f[idx]
            if best_f[idx] > 0.01:
                f_test = best_f.copy()
                f_test[idx] *= 0.9
                score_test = compute_score(f_test)
                if score_test < best_score:
                    best_f = f_test
                    best_score = score_test
                    improved = True
            
            # Try increasing f[idx]
            f_test = best_f.copy()
            f_test[idx] *= 1.1
            score_test = compute_score(f_test)
            if score_test < best_score:
                best_f = f_test
                best_score = score_test
                improved = True
        
        if not improved:
            break
    
    # Ensure nonnegativity
    best_f = np.maximum(0, best_f)
    
    return best_f.tolist()
