"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_objective(x):
    """Compute the upper bound on C3 using FFT for speed."""
    N = len(x)
    # Use FFT to compute autocorrelation
    # Zero pad to 2*N to avoid circular convolution artifacts
    F = np.fft.fft(x, n=2*N)
    power = F * np.conj(F)
    acf = np.fft.ifft(power).real
    
    # Extract non-zero lags
    # Lags 1 to N-1 are at indices 1 to N-1
    # Lags -(N-1) to -1 are at indices N+1 to 2N-1
    non_zero_acf = np.concatenate([acf[1:N], acf[N+1:2*N]])
    
    # The metric from the seed was sum of absolute values
    return np.sum(np.abs(non_zero_acf))

def _generate_chirp(N, freq_scale=1.0):
    """Generate a chirp sequence."""
    t = np.arange(N)
    # Phase is quadratic
    phase = 2 * np.pi * freq_scale * (t**2) / (2 * N)
    s = np.sin(phase)
    # Ensure values are +/- 1
    return np.where(s >= 0, 1, -1)

def _generate_random(N, seed):
    """Generate a random sequence."""
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=N)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try multiple initializations
    best_x = None
    best_obj = np.inf
    
    initializations = []
    # 1. Chirp sequences with different frequencies
    for f in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        initializations.append(_generate_chirp(N, f))
    # 2. Random sequences
    for s in range(10):
        initializations.append(_generate_random(N, s))
        
    for x0 in initializations:
        if time.time() > deadline - 10:
            break
            
        x = x0.copy()
        obj = _compute_objective(x)
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
            
        # Local search
        improved = True
        # Limit iterations per start to avoid getting stuck in one local minimum too long
        max_passes = 100 
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline - 5:
            improved = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Flip bit
                x[i] = -x[i]
                new_obj = _compute_objective(x)
                if new_obj < obj:
                    obj = new_obj
                    improved = True
                    if obj < best_obj:
                        best_obj = obj
                        best_x = x.copy()
                else:
                    # Revert
                    x[i] = -x[i]
                    
    if best_x is None:
        best_x = np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
