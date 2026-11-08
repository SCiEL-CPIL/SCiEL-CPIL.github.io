import numpy as np
import time
import math

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N
    dx = 0.5 / M
    
    f_fine = np.repeat(signed_step, M // N)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _compute_c3_fast(signed_step):
    """
    Faster computation using direct FFT without repeating.
    Since the step function is constant on each interval, we can work with the N-point
    representation and scale appropriately.
    
    Actually, let's think about this more carefully. The autoconvolution of a step function
    with N intervals on [-1/4, 1/4] can be computed by considering the piecewise constant
    nature. But for speed, let's use a smaller FFT.
    
    Key insight: if f is constant on each of N intervals of width w = 0.5/N, then
    (f*f)(t) = integral f(s)f(t-s) ds.
    
    Let's index intervals 0..N-1. Interval i covers [x_i, x_i + w) where x_i = -1/4 + i*w.
    
    The autoconvolution at time t can be computed by summing over pairs of intervals.
    But this is O(N^2) which for N=400 is 160000 operations - might be fast enough.
    
    Actually, let's just use the FFT approach but with a smaller size.
    """
    # Use the direct N-point approach with proper scaling
    # The function values are signed_step (length N), each interval has width w = 0.5/N
    # The autoconvolution in discrete sense:
    # (f*f)[k] = sum_j f[j] * f[(k-j) mod N] * w  ... no, this isn't quite right for continuous
    
    # Let's stick with the FFT approach but optimize it
    M = 4 * N
    dx = 0.5 / M
    
    # Instead of repeating, we can use the fact that f is piecewise constant
    # Create the fine grid more efficiently
    f_fine = np.repeat(signed_step, M // N)
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    np.random.seed(42)
    
    # Simulated annealing parameters
    T_initial = 0.01
    T_final = 0.0001
    cooling_rate = 0.999
    
    # Initialize with all ones (the seed)
    current_step = np.ones(N, dtype=np.int8)
    current_c3 = _compute_c3(current_step)
    
    best_step = current_step.copy()
    best_c3 = current_c3
    
    T = T_initial
    
    iteration = 0
    # Precompute for faster evaluation
    # Let's try a different approach: use a smaller FFT by exploiting the structure
    
    while True:
        # Check deadline with safety margin
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2.0:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        current_step[idx] = -current_step[idx]
        
        new_c3 = _compute_c3(current_step)
        delta_c3 = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_c3 < 0:
            # Always accept improvements
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Probabilistic acceptance for uphill moves
            if T > 1e-10:
                accept_prob = math.exp(-delta_c3 / T)
                if np.random.random() < accept_prob:
                    current_c3 = new_c3
                else:
                    # Reject: revert the flip
                    current_step[idx] = -current_step[idx]
            else:
                # At very low temperature, only accept improvements
                current_step[idx] = -current_step[idx]
        
        # Cool down
        T *= cooling_rate
        iteration += 1
    
    return best_step
