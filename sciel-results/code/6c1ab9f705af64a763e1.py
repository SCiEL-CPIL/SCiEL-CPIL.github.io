"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_bound(signed_step):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    The autoconvolution of a step function with itself, when restricted to
    certain lags, gives an upper bound on the third autocorrelation coefficient.
    We use FFT to efficiently compute this.
    """
    n = len(signed_step)
    # Pad to power of 2 for FFT efficiency
    size = 1
    while size < 2 * n:
        size *= 2
    
    # Compute autocorrelation via FFT
    f = np.zeros(size, dtype=np.complex128)
    f[:n] = signed_step
    F = np.fft.fft(f)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    
    # The relevant lags for the C3 bound: we look at off-peak values
    # For the third autocorrelation inequality, we care about specific lag ranges
    # The bound involves the maximum absolute value of autocorrelation at non-zero lags
    # within certain ranges determined by the interval structure
    
    # For 400 intervals on [-1/4, 1/4], the critical lags are those corresponding
    # to the overlap regions that affect the C3 coefficient
    # We take the max absolute autocorrelation at lags 1 through n-1
    off_peak = autocorr[1:n]
    return np.max(np.abs(off_peak))


def _flip_and_evaluate(signed_step, idx):
    """Flip sign at index idx and return new bound value."""
    new_step = signed_step.copy()
    new_step[idx] = -new_step[idx]
    return _autoconv_bound(new_step), new_step


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Initialize with random signs
    rng = np.random.default_rng(42)
    current = rng.choice([-1.0, 1.0], size=N)
    current_bound = _autoconv_bound(current)
    
    best = current.copy()
    best_bound = current_bound
    
    # Simulated annealing / Metropolis-Hastings parameters
    T_start = 1.0
    T_end = 0.001
    n_iterations = 50000
    
    # Run Metropolis-Hastings
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Temperature schedule (exponential cooling)
        T = T_start * (T_end / T_start) ** (it / n_iterations)
        
        # Choose a random position to flip
        idx = rng.integers(0, N)
        
        # Evaluate the flip
        new_bound, new_step = _flip_and_evaluate(current, idx)
        
        # Metropolis acceptance criterion
        delta = new_bound - current_bound
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = new_step
            current_bound = new_bound
            
            # Update best
            if current_bound < best_bound:
                best = current.copy()
                best_bound = current_bound
    
    # Final greedy polish pass
    if time.time() < deadline - 5.0:
        for _ in range(5):
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for idx in range(N):
                    if time.time() > deadline - 2.0:
                        break
                    new_bound, new_step = _flip_and_evaluate(best, idx)
                    if new_bound < best_bound:
                        best = new_step
                        best_bound = new_bound
                        improved = True
    
    return best
# EVOLVE-BLOCK-END
