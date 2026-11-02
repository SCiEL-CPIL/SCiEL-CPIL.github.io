"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _autoconv_fft(x):
    """Compute full autoconvolution using FFT."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, size)
    F2 = F * F
    conv = np.fft.irfft(F2, size)[:2 * n - 1]
    return conv


def _objective(x):
    """Compute C3' = max|autoconv| / (sum(x))^2"""
    conv = _autoconv_fft(x)
    max_abs = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-10:
        return float('inf')
    return max_abs / (integral * integral)


def _check_deadline(start_time, margin=5.0):
    """Check if we should stop due to time limit."""
    elapsed = time.time() - start_time
    return elapsed > 115.0 - margin


def _try_flip(x, i, start_time):
    """Try flipping sign at position i, return new x and improvement if better."""
    x_new = x.copy()
    x_new[i] = -x_new[i]
    obj_new = _objective(x_new)
    return x_new, obj_new


def _structured_initial(n):
    """Create a structured initial sequence with alternating blocks."""
    x = np.zeros(n)
    # Try different block patterns
    best_x = None
    best_obj = float('inf')
    
    # Pattern 1: alternating +1/-1
    x1 = np.ones(n)
    x1[1::2] = -1
    obj1 = _objective(x1)
    if obj1 < best_obj:
        best_obj = obj1
        best_x = x1.copy()
    
    # Pattern 2: blocks of 2
    x2 = np.ones(n)
    for i in range(0, n, 4):
        x2[i:i+2] = -1
    obj2 = _objective(x2)
    if obj2 < best_obj:
        best_obj = obj2
        best_x = x2.copy()
    
    # Pattern 3: blocks of 4
    x3 = np.ones(n)
    for i in range(0, n, 8):
        x3[i:i+4] = -1
    obj3 = _objective(x3)
    if obj3 < best_obj:
        best_obj = obj3
        best_x = x3.copy()
    
    # Pattern 4: random
    np.random.seed(42)
    x4 = np.random.choice([-1, 1], n)
    obj4 = _objective(x4)
    if obj4 < best_obj:
        best_obj = obj4
        best_x = x4.copy()
    
    return best_x, best_obj


def _local_search(x, start_time, max_iters=10000):
    """Local search by flipping individual positions."""
    current_obj = _objective(x)
    best_x = x.copy()
    best_obj = current_obj
    iters = 0
    
    while iters < max_iters and not _check_deadline(start_time):
        iters += 1
        improved = False
        # Try random flips
        for _ in range(10):
            i = np.random.randint(N)
            x_new, obj_new = _try_flip(x, i, start_time)
            if obj_new < best_obj - 1e-12:
                x = x_new
                best_obj = obj_new
                best_x = x_new.copy()
                improved = True
                break
        
        if not improved:
            # Try systematic sweep
            for i in range(N):
                if _check_deadline(start_time):
                    break
                x_new, obj_new = _try_flip(x, i, start_time)
                if obj_new < best_obj - 1e-12:
                    x = x_new
                    best_obj = obj_new
                    best_x = x_new.copy()
                    improved = True
    
    return best_x, best_obj


def construct():
    """Construct signed step heights minimizing C3'."""
    start_time = time.time()
    
    # Initialize with structured patterns
    x, obj = _structured_initial(N)
    best_x = x.copy()
    best_obj = obj
    
    # Local search refinement
    best_x, best_obj = _local_search(x, start_time, max_iters=5000)
    
    # If time permits, try more iterations
    if not _check_deadline(start_time, margin=10.0):
        best_x, best_obj = _local_search(best_x, start_time, max_iters=10000)
    
    return best_x
# EVOLVE-BLOCK-END
