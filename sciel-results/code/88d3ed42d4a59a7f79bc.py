import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_acf_fft(h):
    """Compute autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _max_sidelobe(acf):
    """Compute max |autocorrelation| at non-zero lags."""
    return np.max(np.abs(acf[1:]))

def _local_search(h, deadline):
    """
    Iterated Local Search with single-bit flips.
    """
    best_h = h.copy()
    best_acf = _compute_acf_fft(best_h)
    best_val = _max_sidelobe(best_acf)
    
    current_h = best_h.copy()
    current_acf = best_acf.copy()
    current_val = best_val
    
    # Simple hill climbing with random restarts/perturbations
    stagnation_count = 0
    max_stagnation = 1000
    
    while time.time() < deadline:
        # Try to improve
        improved = False
        # Randomize the order of indices to break symmetry
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Flip
            current_h[i] = -current_h[i]
            
            # Recompute ACF (FFT is fast enough for N=400)
            new_acf = _compute_acf_fft(current_h)
            new_val = _max_sidelobe(new_acf)
            
            if new_val < current_val:
                current_val = new_val
                current_acf = new_acf
                improved = True
                stagnation_count = 0
                if new_val < best_val:
                    best_val = new_val
                    best_h = current_h.copy()
            else:
                # Revert
                current_h[i] = -current_h[i]
        
        if not improved:
            stagnation_count += 1
            if stagnation_count > max_stagnation:
                # Perturb: flip a random block of size 5
                k = 5
                start = np.random.randint(0, N - k)
                current_h[start:start+k] = -current_h[start:start+k]
                current_acf = _compute_acf_fft(current_h)
                current_val = _max_sidelobe(current_acf)
                stagnation_count = 0
        
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    np.random.seed(42)
    
    # Multi-start
    best_h_overall = None
    best_val_overall = np.inf
    
    # Run a few restarts
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        h_init = np.random.choice([-1, 1], size=N).astype(np.float64)
        
        # Allocate remaining time to this restart
        restart_deadline = time.time() + time_per_restart * 1.1 # Slight overlap allowed
        
        h_res = _local_search(h_init, restart_deadline)
        
        acf_res = _compute_acf_fft(h_res)
        val_res = _max_sidelobe(acf_res)
        
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_h_overall = h_res.copy()
            
        # Check if we have time for next restart
        if time.time() > deadline - 5:
            break

    return best_h_overall
# EVOLVE-BLOCK-END
