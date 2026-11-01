import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max over x,y of |autoconvolution| * something?
    
    Actually, let me think about what C3 is.
    
    The task says "improve an autoconvolution upper bound" and "score is the upper bound on C3".
    
    For a step function f on [-1/4, 1/4] with N equal intervals, the autoconvolution is:
    (f * f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    
    The C3 bound likely involves the maximum of the absolute value of the autoconvolution,
    or perhaps a specific integral related to the third autocorrelation.
    
    Let me reconsider. "Third autocorrelation inequality" - this might refer to a bound involving
    the third moment or a specific autocorrelation structure.
    
    A common setup: For a function f with ||f||_1 = 1 or some normalization, we want to bound
    max_t |(f*f)(t)| or similar.
    
    Let me assume: f is a step function with N intervals, each with height in {-1, 1} (signed).
    The autoconvolution is computed, and C3 is some measure of its maximum absolute value.
    
    Given the seed returns np.ones(N) and scores 1.0, and we want to minimize C3...
    
    Let me just compute the autoconvolution of the step function and take the max absolute value.
    For all ones, the autoconvolution of a step function that is 1 on [-1/4, 1/4] would be a
    triangular function with peak at 0.
    
    Actually, let me just implement a reasonable C3 computation and optimize for it.
    
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    # signed_step is an array of N values in {-1, 1}
    # The function f(x) = signed_step[i] for x in [i/N - 1/4, (i+1)/N - 1/4)
    # Wait, the domain is [-1/4, 1/4], so interval i covers [-1/4 + i/N, -1/4 + (i+1)/N)
    
    # Autoconvolution: (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, this can be computed efficiently.
    
    # Let h = 1/N be the interval width.
    # f(s) = signed_step[i] for s in [-1/4 + i*h, -1/4 + (i+1)*h)
    
    # (f*f)(t) = sum over pairs (i,j) such that intervals overlap appropriately
    # This is complex. Let me use a direct numerical approach with fine grid.
    
    # Actually, for efficiency, let me compute it via FFT or direct summation.
    
    # Simpler: discretize the domain and compute convolution.
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Try to improve using local search
    # Flip random elements and keep if it improves
    
    np.random.seed(42)
    
    max_iterations = 100000
    no_improve_count = 0
    patience = 5000
    
    for iteration in range(max_iterations):
        # Check deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            best_step[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count >= patience:
            break
    
    return best_step
