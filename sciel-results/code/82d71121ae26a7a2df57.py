"""Find flattest +/-1 polynomial with 24 coefficients (min sup norm)."""
import numpy as np
import time
import random

def _eval_supnorm(coeffs, M=32768):
    """Compute max |P(z)| for |z|=1 using dense FFT sampling."""
    c = np.zeros(M, dtype=np.complex128)
    c[:24] = coeffs
    F = np.fft.fft(c)
    return float(np.max(np.abs(F)))

def _refine_peak(coeffs, theta0, M=32768):
    """Golden-section refinement around theta0 to find local max of |P(e^{i theta})|."""
    c_arr = np.array(coeffs, dtype=np.complex128)
    
    def f(theta):
        z = np.exp(1j * theta)
        # Horner's method
        val = 0.0 + 0.0j
        for ck in reversed(c_arr):
            val = val * z + ck
        return abs(val)
    
    # Narrow window
    delta = 2.0 * np.pi / M
    a, b = theta0 - delta, theta0 + delta
    
    # Golden section
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c_pt = b - gr * (b - a)
    d_pt = a + gr * (b - a)
    fc = f(c_pt)
    fd = f(d_pt)
    
    for _ in range(40):
        if fc < fd:
            a = c_pt
            c_pt = d_pt
            fc = fd
            d_pt = a + gr * (b - a)
            fd = f(d_pt)
        else:
            b = d_pt
            d_pt = c_pt
            fd = fc
            c_pt = b - gr * (b - a)
            fc = f(c_pt)
    
    mid = (a + b) / 2.0
    return f(mid)

def _full_supnorm(coeffs):
    """Compute sup norm with peak refinement for accuracy ~1e-9."""
    M = 32768
    c = np.zeros(M, dtype=np.complex128)
    c[:24] = coeffs
    F = np.fft.fft(c)
    absF = np.abs(F)
    
    # Find top peaks
    top_k = 10
    peak_indices = np.argsort(absF)[-top_k:]
    peak_indices = peak_indices[np.argsort(peak_indices)]  # sort by angle
    
    max_val = 0.0
    for idx in peak_indices:
        theta = 2.0 * np.pi * idx / M
        val = _refine_peak(coeffs, theta, M)
        if val > max_val:
            max_val = val
    
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    N = 24
    sqrtN = np.sqrt(float(N))
    
    # Initial: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N], dtype=np.complex128)
    best_val = _eval_supnorm(best_coeffs)
    
    # Simulated annealing with FFT-based evaluation for speed
    M = 32768
    c_padded = np.zeros(M, dtype=np.complex128)
    c_padded[:N] = best_coeffs
    
    # Precompute FFT of current
    F = np.fft.fft(c_padded)
    current_val = float(np.max(np.abs(F)))
    
    # Track best
    best_coeffs_arr = best_coeffs.copy()
    best_val_exact = current_val  # Will refine later
    
    # Simulated annealing parameters
    T = 2.0
    T_min = 0.01
    cooling = 0.9997
    num_coeffs = N
    
    random.seed(42)
    np.random.seed(42)
    
    while time.time() < deadline:
        # Update temperature
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Pick a random position to flip
        idx = random.randint(0, num_coeffs - 1)
        
        # Compute change in FFT when flipping coefficient at idx
        # c[idx] goes from c[idx] to -c[idx], change is -2*c[idx]
        delta_c = -2.0 * best_coeffs_arr[idx]
        
        # FFT of the delta: delta_c * exp(2*pi*i*idx*k/M) for k=0..M-1
        # This is delta_c * twiddle[idx, :]
        # We can compute this efficiently
        k = np.arange(M)
        twiddle = np.exp(2j * np.pi * idx * k / M)
        delta_F = delta_c * twiddle
        
        new_F = F + delta_F
        new_val = float(np.max(np.abs(new_F)))
        
        # Accept or reject
        if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / T):
            # Accept
            best_coeffs_arr[idx] = -best_coeffs_arr[idx]
            F = new_F
            current_val = new_val
            
            if current_val < best_val_exact:
                best_val_exact = current_val
                best_coeffs_arr_copy = best_coeffs_arr.copy()
        else:
            # Reject, no change
            pass

        # Periodic check with refined sup norm for the current best
        if int((time.time() - start_time) * 10) % 5 == 0 and current_val < best_val_exact + 0.01:
            # This is a bit expensive, do it less frequently
            pass
    
    # Final refinement of the best found
    # Convert back to real +/-1 array for refinement
    best_real = np.zeros(N, dtype=np.float64)
    for i in range(N):
        best_real[i] = 1.0 if abs(best_coeffs_arr[i] - 1.0) < 0.1 else -1.0
    
    # Use the full refined sup norm on the best candidate
    final_val = _full_supnorm(best_real)
    
    # Return as list of +/-1
    result = [1 if x > 0 else -1 for x in best_real]
    return result
