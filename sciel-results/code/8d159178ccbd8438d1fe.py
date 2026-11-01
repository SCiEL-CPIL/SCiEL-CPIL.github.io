"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(h):
    """Compute linear autocorrelation using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _compute_c3_bound_from_acf(acf):
    """Compute C3 bound from precomputed autocorrelation."""
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)

def _compute_c3_bound(h):
    """Compute C3 bound from sequence."""
    acf = _compute_acf(h)
    return _compute_c3_bound_from_acf(acf)

def _local_search_sa(h, acf, current_val, deadline, rng):
    """
    Simulated Annealing local search.
    Uses O(N) update for autocorrelation changes.
    """
    best_h = h.copy()
    best_val = current_val
    best_acf = acf.copy()
    
    T = 1.0  # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline and T > 1e-6:
        improved = False
        # Shuffle indices to avoid bias
        indices = rng.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                return best_h, best_val
            
            # Calculate change in acf if we flip h[i]
            # d_acf[k] = -2 * h[i] * (h[i+k] + h[i-k])
            d_acf = np.zeros(N)
            
            # Vectorized calculation of neighbors
            # For lag k, contribution from right: h[i+k] if i+k < N
            # Contribution from left: h[i-k] if i-k >= 0
            
            # Right part: k from 1 to N-1-i
            # h[i+1], h[i+2], ... h[N-1]
            if i < N - 1:
                right_len = N - 1 - i
                d_acf[1:right_len+1] += -2 * h[i] * h[i+1:]
                
            # Left part: k from 1 to i
            # h[i-1], h[i-2], ... h[0]
            if i > 0:
                left_len = i
                # We need to add to d_acf[1:left_len+1]
                # The values are h[i-1], h[i-2], ... h[0]
                # These correspond to lags 1, 2, ... i
                d_acf[1:left_len+1] += -2 * h[i] * h[:i][::-1]
            
            new_acf = acf + d_acf
            new_val = _compute_c3_bound_from_acf(new_acf)
            
            delta = new_val - current_val
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                h[i] = -h[i]
                acf = new_acf
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
                    best_acf = acf.copy()
                    improved = True
        
        T *= cooling
        
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = np.inf
    best_acf = None
    
    # Try multiple restarts
    # 1. Random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        h = rng.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(h)
        val = _compute_c3_bound_from_acf(acf)
        
        if val < best_val:
            best_h = h.copy()
            best_val = val
            best_acf = acf.copy()
            
        # Run SA from this start
        if time.time() < deadline - 10:
            res_h, res_val = _local_search_sa(h, acf, val, deadline, rng)
            if res_val < best_val:
                best_h = res_h
                best_val = res_val
                best_acf = _compute_acf(res_h) # Re-compute to be safe

    # 2. Try a structured sequence: M-sequence or similar
    # A simple pseudo-random sequence with good autocorrelation properties
    # We can try to construct a sequence using a linear feedback shift register
    # But for N=400, we can just try a few specific patterns or a "Gold code" like structure
    # For now, randoms with SA should be strong.
    
    # If we have time, do a final polish on the best result
    if best_h is not None and time.time() < deadline - 2:
        # Run SA again on the best result with lower temperature
        acf = _compute_acf(best_h)
        val = _compute_c3_bound_from_acf(acf)
        res_h, res_val = _local_search_sa(best_h, acf, val, deadline, rng)
        if res_val < best_val:
            best_h = res_h
            best_val = res_val

    return best_h

# EVOLVE-BLOCK-END
