"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_metric(x):
    """
    Computes the maximum absolute autocorrelation value at non-zero lags.
    This serves as the proxy for the 'upper bound on C3' where lower is better.
    """
    # Ensure x is +/-1
    # x = np.sign(x)
    # x[x == 0] = 1
    
    # FFT for fast autocorrelation
    # Pad to next power of 2 or just use 2*N to avoid circular wrap-around for lags up to N-1
    # For N=400, 2*N=800 is not a power of 2, but FFT is fast enough. 
    # Actually, for linear autocorrelation of length N, we need FFT size >= 2N-1.
    # Let's use size 2*N for simplicity and speed (slightly more than needed).
    fft_size = 2 * N
    X = np.fft.fft(x, n=fft_size)
    power = np.abs(X) ** 2
    acf = np.fft.ifft(power).real
    
    # The autocorrelation at lag 0 is acf[0] = sum(x^2) = N.
    # Lags 1 to N-1 are at acf[1:N].
    # Lags -1 to -(N-1) are at acf[fft_size-N+1 : fft_size].
    # Due to symmetry for real signals, acf[k] = acf[-k].
    # So we only need to check acf[1:N].
    
    # Extract non-zero lags
    sidelobes = acf[1:N]
    
    # Return the maximum absolute value
    return np.max(np.abs(sidelobes))

def _simulated_annealing():
    """
    Uses Simulated Annealing to minimize the max autocorrelation sidelobe.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Initial random sequence
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    
    current_score = _autocorr_metric(x)
    best_x = x.copy()
    best_score = current_score
    
    # Annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # We will perform iterations until time limit.
    # Estimate iterations per second to adjust if necessary, but for now fixed loop with time check.
    
    T = T_start
    i = 0
    while time.time() < deadline:
        i += 1
        
        # Generate a neighbor by flipping one bit
        # Randomly select an index to flip
        idx = rng.integers(0, N)
        x[idx] *= -1
        
        new_score = _autocorr_metric(x)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Reject move, revert bit
            x[idx] *= -1
            
        # Cool down
        T = max(T * cooling_rate, T_end)
        
        # Optional: If stuck for too long, reheat? 
        # For simplicity, we just let it cool. If T hits T_end, we can restart or just continue with small T.
        # Given the time limit, simple cooling is a good structural change from the previous spectral shaping.
        
    return best_x

def construct():
    """
    Constructs the signed step function.
    Returns a sequence of 400 values, each -1 or 1.
    """
    x = _simulated_annealing()
    return x
# EVOLVE-BLOCK-END
