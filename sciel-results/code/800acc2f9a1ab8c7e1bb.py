"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_peak_autocorr(h):
    """Compute the maximum absolute value of the linear autocorrelation at non-zero lags."""
    n = len(h)
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    auto = np.fft.irfft(power, n=fft_size)
    
    # auto[0] is r[0] = sum(h^2) = N
    # We want max |auto[k]| for k=1..n-1
    # Note: auto[k] for k>0 corresponds to r[k]
    # Due to circular correlation properties in FFT, we take the first n elements
    # auto[1:n] contains r[1]...r[n-1]
    
    peak = np.max(np.abs(auto[1:n]))
    return peak

def _local_search(h, time_limit):
    """Local search to minimize peak autocorrelation."""
    n = len(h)
    h = h.copy()
    current_peak = _compute_peak_autocorr(h)
    best_peak = current_peak
    best_h = h.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500 # Allow more iterations before stopping
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle indices to avoid bias towards early indices
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            
            # Try flipping bit i
            h[i] = -h[i]
            new_peak = _compute_peak_autocorr(h)
            
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
                no_improve_count = 0
                # Optional: break to restart the scan from the beginning
                # Or continue to find more improvements in the same pass
                # Breaking is often better for escaping local minima in this metric
                break
            else:
                h[i] = -h[i] # Revert
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_h, best_peak

def _thue_morse_init(n):
    """Generate Thue-Morse sequence of length n."""
    h = np.zeros(n)
    for i in range(n):
        # Count number of 1s in binary representation of i
        # Thue-Morse is (-1)^(popcount(i))
        # h[i] = 1 if even number of 1s, -1 if odd
        count = 0
        temp = i
        while temp:
            count += temp & 1
            temp >>= 1
        h[i] = 1.0 if count % 2 == 0 else -1.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_h = None
    best_peak = np.inf
    
    # 1. Thue-Morse Initialization
    h_tm = _thue_morse_init(N)
    peak_tm = _compute_peak_autocorr(h_tm)
    if peak_tm < best_peak:
        best_peak = peak_tm
        best_h = h_tm.copy()
    
    # 2. Random Initialization with Local Search
    # Use multiple random restarts
    n_restarts = 10
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Random signs
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        
        # Run local search
        remaining = deadline - time.time() - 2.0
        if remaining <= 0:
            break
        time_limit = min(remaining, time_per_restart)
        
        h_opt, peak_opt = _local_search(h_rand, time_limit)
        
        if peak_opt < best_peak:
            best_peak = peak_opt
            best_h = h_opt
    
    # 3. Final polish with remaining time
    if best_h is not None and time.time() < deadline - 1.0:
        h_final, peak_final = _local_search(best_h, deadline - time.time() - 0.5)
        if peak_final < best_peak:
            best_peak = peak_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
