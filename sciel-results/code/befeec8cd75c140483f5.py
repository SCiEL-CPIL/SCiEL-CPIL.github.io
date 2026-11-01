"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_bound(h):
    """
    Compute the C3 upper bound for a signed step function with N intervals.
    
    The third autocorrelation inequality relates the autoconvolution to the 
    L2 norm. For a step function f with values h_i on intervals of width dx,
    the autoconvolution at lag 0 is sum(h_i^2 * dx) = ||h||_2^2 * dx.
    
    The 'upper bound on C3' we want to minimize is related to the maximum 
    sidelobe of the autoconvolution relative to the peak, or a related 
    measure. Based on the task description and typical autocorrelation 
    inequality problems, we interpret this as minimizing the maximum absolute 
    value of the autoconvolution away from the origin (sidelobes), or a 
    normalized measure.
    
    However, the task says "improve an autoconvolution upper bound" and 
    "score is the upper bound on C3". A common measure in such contexts is 
    the ratio of the maximum sidelobe to the main lobe, or simply the maximum 
    sidelobe value. Let's use the maximum absolute value of the discrete 
    autoconvolution excluding the zero-lag term. This is a standard measure 
    of autocorrelation quality (lower sidelobes are better).
    
    We compute the circular autoconvolution (or linear, but circular is faster 
    with FFT). For a step function, the autoconvolution at lag k is 
    sum_i h_i * h_{i+k}. We want to minimize max_{k != 0} |sum_i h_i * h_{i+k}|.
    
    To make the problem well-defined and scale-invariant, we normalize h such 
    that ||h||_2^2 = 1. Then the zero-lag autoconvolution is 1. We minimize 
    the maximum sidelobe.
    """
    # Normalize to unit L2 norm
    norm = np.sqrt(np.sum(h**2))
    if norm < 1e-12:
        return np.inf
    h_norm = h / norm
    
    # Compute autoconvolution via FFT
    # Use a larger FFT size to avoid circular convolution artifacts if we 
    # wanted linear, but for step functions on a fixed domain, circular is 
    # often the intended interpretation for periodic-like signals, or we 
    # can use linear. Let's use linear autoconvolution to be safe.
    # Actually, for N=400, linear autoconvolution has length 2N-1.
    # We can compute it via FFT efficiently.
    fft_size = 2 * N
    h_fft = np.fft.rfft(h_norm, n=fft_size)
    ac_fft = h_fft * np.conj(h_fft)
    ac = np.fft.irfft(ac_fft, n=fft_size)
    
    # The autoconvolution is symmetric. The zero-lag is at index 0 and also 
    # appears at the end due to FFT. The linear autoconvolution of length 2N-1 
    # is centered. Let's extract the relevant part.
    # Actually, irfft gives the full linear convolution if we zero-pad correctly.
    # For autoconvolution of h_norm (length N) with itself, the result has 
    # length 2N-1. The FFT approach with size 2N gives the linear convolution 
    # in the first N and last N-1 elements, but due to symmetry, we can just 
    # take the first N elements (lags 0 to N-1) and the rest are mirrored.
    # The zero-lag is ac[0]. Sidelobes are ac[1:N] and ac[N:2N-1] (but these 
    # are the same as ac[N-1:0:-1]).
    
    # To be precise, the linear autoconvolution c[k] = sum_i h[i] h[i+k] for 
    # k = -(N-1) to N-1. The FFT gives us c[0] to c[2N-2] in a circular 
    # manner if not padded. With padding to 2N, we get c[0] to c[2N-1], 
    # where c[0] is the zero-lag, c[1] to c[N-1] are positive lags, and 
    # c[N] to c[2N-1] are negative lags (mirrored).
    
    # The maximum sidelobe is the max of |c[k]| for k != 0.
    # Due to symmetry, we only need to check k = 1 to N-1.
    if N > 1:
        max_sidelobe = np.max(np.abs(ac[1:N]))
    else:
        max_sidelobe = 0.0
    
    return max_sidelobe

def _objective(h):
    """Objective function to minimize."""
    return _compute_bound(h)

def construct():
    """
    Construct a signed step function with N intervals to minimize the 
    autoconvolution upper bound (max sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Initialize with random signs
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    best_score = _objective(best_h)
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    
    current_h = h.copy()
    current_score = best_score
    
    step = 0
    while time.time() < deadline:
        # Flip a random sign
        idx = np.random.randint(N)
        old_h = current_h[idx].copy()
        current_h[idx] = -current_h[idx]
        new_score = _objective(current_h)
        
        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            current_h[idx] = old_h
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with higher temperature if stuck
            T = 0.1
        
        step += 1
        # Periodic restart to escape local minima
        if step % 1000 == 0:
            # Try a random restart with a few flips
            for _ in range(5):
                idx = np.random.randint(N)
                current_h[idx] = -current_h[idx]
            new_score = _objective(current_h)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
