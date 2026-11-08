"""Fourier-domain optimization for Erdos minimum overlap."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill

def _fourier_to_time(hat_h):
    """
    Convert Fourier coefficients to time domain h.
    """
    return np.fft.ifft(hat_h).real

def _objective_fourier(x):
    """
    x contains the free Fourier coefficients.
    Structure of x:
    - Real and Imag parts of hat_h[m] for m = 1 to N//2 - 1.
    - Real part of hat_h[N//2] (Nyquist, if N even).
    - hat_h[0] is fixed to N/2.
    - hat_h[-m] = conj(hat_h[m]).
    
    Objective: Max overlap + penalties for h bounds.
    """
    # Reconstruct hat_h
    hat_h = np.zeros(N, dtype=complex)
    hat_h[0] = N / 2.0
    
    idx = 0
    for m in range(1, N // 2):
        hat_h[m] = x[idx] + 1j * x[idx + 1]
        hat_h[N - m] = np.conj(hat_h[m])
        idx += 2
    
    if N % 2 == 0:
        hat_h[N // 2] = x[idx] + 0j
        
    h = _fourier_to_time(hat_h)
    
    # Compute max overlap
    # overlap[k] = sum_i h[i] (1 - h[(i+k)%N])
    # = sum_i h[i] - sum_i h[i] h[(i+k)%N]
    # sum_i h[i] = N/2
    # autocorr[k] = sum_i h[i] h[(i+k)%N] = ifft(|hat_h|^2)[k]
    autocorr = np.fft.ifft(np.abs(hat_h)**2).real
    overlap = N/2.0 - autocorr
    max_ov = np.max(overlap)
    
    # Penalties for h bounds [0, 1]
    h_min = np.min(h)
    h_max = np.max(h)
    pen_low = 0.0
    pen_high = 0.0
    if h_min < 0:
        pen_low = 1000.0 * (h_min ** 2)
    if h_max > 1:
        pen_high = 1000.0 * ((h_max - 1.0) ** 2)
        
    return max_ov + pen_low + pen_high

def construct():
    start_time = time.time()
    
    # Number of free variables
    n_free = 0
    for m in range(1, N // 2):
        n_free += 2
    if N % 2 == 0:
        n_free += 1
        
    x0 = np.zeros(n_free)
    
    best_x = x0.copy()
    best_val = _objective_fourier(x0)
    
    np.random.seed(42)
    
    # Multi-start optimization
    for restart in range(20):
        if time.time() > DEADLINE - 10:
            break
            
        if restart == 0:
            x_init = x0.copy()
        else:
            # Random perturbation in Fourier domain
            scale = 0.1 + 0.05 * (restart % 5)
            x_init = x0 + np.random.randn(n_free) * scale
            
        result = minimize(
            _objective_fourier,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        val = _objective_fourier(result.x)
        if val < best_val:
            best_val = val
            best_x = result.x.copy()
            
        if time.time() > DEADLINE - 20:
            break
            
    # Convert best Fourier coefficients to time domain
    hat_h = np.zeros(N, dtype=complex)
    hat_h[0] = N / 2.0
    idx = 0
    for m in range(1, N // 2):
        hat_h[m] = best_x[idx] + 1j * best_x[idx + 1]
        hat_h[N - m] = np.conj(hat_h[m])
        idx += 2
    if N % 2 == 0:
        hat_h[N // 2] = best_x[idx] + 0j
        
    h = _fourier_to_time(hat_h)
    
    # Clip and adjust sum to be safe
    h = np.clip(h, 0, 1)
    
    # Adjust sum to exactly N/2
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        for _ in range(10):
            diff = N/2.0 - np.sum(h)
            if abs(diff) < 1e-10:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
            
    return h
