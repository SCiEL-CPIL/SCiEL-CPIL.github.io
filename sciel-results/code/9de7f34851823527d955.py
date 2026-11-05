"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autocorr_side_lobs(f):
    """
    Compute the autocorrelation side-lobes (lags 1 to N-1) efficiently.
    Uses FFT for speed: O(N log N) instead of O(N^2).
    """
    n = len(f)
    # Zero-pad to 2n to avoid circular convolution effects
    f_padded = np.zeros(2 * n)
    f_padded[:n] = f
    F = np.fft.fft(f_padded)
    acf = np.fft.ifft(F * np.conj(F)).real[:n]
    return acf[1:]

def _score_psl(side_lobs):
    """Peak Sidelobe Level: max absolute value of side-lobes."""
    if len(side_lobs) == 0:
        return 0.0
    return np.max(np.abs(side_lobs))

def _sa_search(f_init, time_limit, rng):
    """
    Simulated Annealing to minimize PSL of autocorrelation side-lobes.
    Constraint: sum(f) != 0.
    """
    n = len(f_init)
    f = f_init.copy()
    current_side = _autocorr_side_lobs(f)
    current_score = _score_psl(current_side)
    
    best_f = f.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9995 # Cooling rate
    T = T_start
    
    start_time = time.time()
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random index to flip
        k = rng.randint(n)
        
        # Check constraint: new sum must not be 0
        current_sum = np.sum(f)
        # delta_sum is -2 * f[k]
        new_sum = current_sum - 2 * f[k]
        
        if new_sum == 0:
            # Reject move if it violates constraint
            T *= alpha
            continue
            
        # Flip
        f[k] = -f[k]
        
        # Calculate new score
        # For speed, we could update autocorr incrementally, but FFT is fast enough for N=400
        # O(N log N) ~ 400 * 9 = 3600 ops. Very fast.
        new_side = _autocorr_side_lobs(f)
        new_score = _score_psl(new_side)
        
        delta_E = new_score - current_score
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            # Accept
            current_score = new_score
            current_side = new_side
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            # Reject
            f[k] = -f[k] # Revert
        
        T *= alpha
        
    return best_f, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_f = None
    best_score = np.inf
    
    # Multiple random restarts
    max_restarts = 20
    time_per_restart = (deadline - start_time) / max_restarts
    
    for i in range(max_restarts):
        if time.time() >= deadline - 2.0:
            break
            
        # Generate random initial sequence
        # Ensure sum is not zero by construction (e.g., start with random, if sum 0, flip one bit)
        h = rng.choice([-1.0, 1.0], size=N)
        if np.sum(h) == 0:
            h[0] = -h[0] # Flip first bit to ensure non-zero sum
        
        # Run SA
        remaining_time = deadline - time.time() - 1.0
        if remaining_time <= 0:
            break
        
        h_opt, score_opt = _sa_search(h, remaining_time * 0.9, rng)
        
        if score_opt < best_score:
            best_score = score_opt
            best_f = h_opt
            
    if best_f is None:
        # Fallback: All 1s
        best_f = np.ones(N)
        
    return best_f

# EVOLVE-BLOCK-END
