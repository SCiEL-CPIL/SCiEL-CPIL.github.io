"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(a):
    """Compute the maximum absolute value of the linear autoconvolution."""
    n = len(a)
    # Use FFT for efficient autoconvolution
    size = 2 * n
    f = np.fft.fft(a, n=size)
    autoconv = np.fft.ifft(f * np.conj(f)).real
    # Linear autoconvolution has 2n-1 values, indices 0 to 2n-2
    # The max absolute value over all lags
    return np.max(np.abs(autoconv[:2*n-1]))

def _vdcorput_signs(n):
    """Generate signs based on van der Corput sequence."""
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Van der Corput in base 2
        x = 0.0
        d = 0.5
        val = i
        while val > 0:
            x += (val % 2) * d
            val //= 2
            d *= 0.5
        # Map to sign: +1 if x < 0.5, -1 otherwise
        signs[i] = 1 if x < 0.5 else -1
    return signs

def _local_search(a, max_time=100.0):
    """Local search: flip bits to reduce max absolute autoconvolution."""
    a = a.copy()
    n = len(a)
    current_val = _autoconv_max_abs(a)
    best_a = a.copy()
    best_val = current_val
    start_time = time.time()
    timeout = start_time + max_time
    
    # Precompute size for FFT
    fft_size = 2 * n
    
    while time.time() < timeout:
        improved = False
        # Try flipping each bit
        for i in range(n):
            if time.time() >= timeout:
                break
            old = a[i]
            a[i] = -old
            val = _autoconv_max_abs(a)
            if val < current_val:
                current_val = val
                if val < best_val:
                    best_val = val
                    best_a = a.copy()
                improved = True
            else:
                a[i] = old
        
        if not improved:
            # Try random restarts within current best neighborhood
            # Or accept a small perturbation to escape local minima
            if best_val < 1e10:  # if we've made progress
                # Random perturbation
                for _ in range(10):
                    idx = np.random.randint(n)
                    a[idx] *= -1
                current_val = _autoconv_max_abs(a)
                if current_val > best_val * 1.1:
                    a = best_a.copy()
                    current_val = best_val
            else:
                break
    
    return best_a, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial sequence: van der Corput based
    a = _vdcorput_signs(N).astype(np.float64)
    
    # Run local search with time budget
    best_a, best_val = _local_search(a, max_time=100.0)
    
    # If we still have time, try a few random restarts
    while time.time() < deadline:
        # Random initial sequence
        a_rand = np.random.choice([-1.0, 1.0], size=N)
        a_rand, val_rand = _local_search(a_rand, max_time=min(5.0, deadline - time.time()))
        if val_rand < best_val:
            best_a = a_rand
            best_val = val_rand
    
    return best_a.astype(np.float64)

# EVOLVE-BLOCK-END
