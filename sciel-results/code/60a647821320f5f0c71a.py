"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _peak_autocorr(s):
    """Compute the peak absolute value of aperiodic autocorrelation for non-zero lags."""
    ac = np.correlate(s, s, mode='full')
    # Exclude the zero-lag term (index N-1)
    # ac[:N-1] are negative lags, ac[N:] are positive lags
    return np.max(np.abs(np.concatenate([ac[:N-1], ac[N:]])))

def _local_search(s, timeout, best_peak, best_s):
    """Perform local search starting from s. Returns (best_peak, best_s)."""
    current_peak = _peak_autocorr(s)
    if current_peak < best_peak:
        best_peak = current_peak
        best_s = s.copy()
        
    improved = True
    while improved and (time.time() < timeout):
        improved = False
        for i in range(N):
            if time.time() > timeout:
                break
            old_val = s[i]
            s[i] = -old_val
            new_peak = _peak_autocorr(s)
            if new_peak < current_peak:
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_s = s.copy()
                improved = True
            else:
                s[i] = old_val
    return best_peak, best_s

def construct():
    start_time = time.time()
    timeout = start_time + 110.0  # Safety margin
    
    best_peak = np.inf
    best_s = np.ones(N)
    
    # Strategy 1: Thue-Morse initialization
    # Generate Thue-Morse sequence
    tm_seq = np.array([bin(i).count('1') % 2 for i in range(N)])
    s_tm = 1 - 2 * tm_seq
    best_peak, best_s = _local_search(s_tm, timeout, best_peak, best_s)
    
    # Strategy 2: Random initializations
    # We can try a few random starts if time permits
    rng = np.random.default_rng(42)
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() > timeout:
            break
        s_rand = rng.choice([-1, 1], size=N)
        best_peak, best_s = _local_search(s_rand, timeout, best_peak, best_s)
        
    return best_s
# EVOLVE-BLOCK-END
