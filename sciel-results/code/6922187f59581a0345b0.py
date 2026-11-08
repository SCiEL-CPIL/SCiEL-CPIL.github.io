import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution
    of the step function with itself.
    """
    # signed_step is an array of N values in {-1, 1}
    # The function f(x) = signed_step[i] for x in [-1/4 + i/N, -1/4 + (i+1)/N)
    
    # Autoconvolution: (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    # For step functions, this can be computed efficiently using the fact that
    # the convolution of two step functions is a piecewise linear function.
    
    # Let h = 1/N be the interval width.
    # f(s) = signed_step[i] for s in [-1/4 + i*h, -1/4 + (i+1)*h)
    
    # The autoconvolution (f*f)(t) for t in [-1/2, 1/2] is:
    # (f*f)(t) = sum_{i,j} signed_step[i] * signed_step[j] * overlap_length(i,j,t)
    # where overlap_length(i,j,t) is the length of the intersection of
    # [i*h, (i+1)*h) and [t-j*h, t-(j+1)*h) (after shifting by -1/4)
    
    # More precisely, if we let the support be [0, 1] (by shifting), then:
    # f(s) = signed_step[i] for s in [i*h, (i+1)*h)
    # (f*f)(t) = sum_{i,j} signed_step[i] * signed_step[j] * max(0, min((i+1)*h, t+j*h) - max(i*h, t+(j-1)*h))
    
    # This is O(N^2) per evaluation, which is too slow for many iterations.
    # Instead, we can use the fact that the autoconvolution of a step function
    # can be computed via FFT on a fine grid.
    
    # Use a fine grid with M = N * K points, where K is a small integer.
    K = 4  # refinement factor
    M = N * K
    dx = 1.0 / M  # domain length is 1 (after shifting)
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = i * K
        end = (i + 1) * K
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-1, 1] in the shifted domain
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with a random signed step function
    np.random.seed(123)
    best_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Use simulated annealing to search for better solutions
    # Start with high temperature and cool down gradually
    
    max_iterations = 50000
    current_temp = 1.0
    cooling_rate = 0.9999
    min_temp = 0.01
    
    for iteration in range(max_iterations):
        # Check deadline
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Try flipping a random element
        idx = np.random.randint(N)
        old_val = best_step[idx]
        best_step[idx] = -old_val
        
        new_c3 = _compute_c3(best_step)
        
        # Accept if better, or with probability exp(-(new_c3 - best_c3) / current_temp)
        if new_c3 < best_c3:
            best_c3 = new_c3
        else:
            delta = new_c3 - best_c3
            if delta > 0 and np.random.rand() < np.exp(-delta / current_temp):
                best_c3 = new_c3
            else:
                best_step[idx] = old_val
        
        # Cool down
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp
    
    return best_step
