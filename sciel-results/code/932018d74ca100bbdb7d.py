import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_max_acf(f):
    """
    Compute the maximum absolute autocorrelation value for non-zero lags.
    This is the primary metric for minimizing peak sidelobe levels.
    """
    # Use FFT for efficient autocorrelation calculation
    # Zero-pad to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    # Power spectrum
    P = F * np.conj(F)
    # Inverse FFT to get autocorrelation
    # The result is circular autocorrelation. We need linear autocorrelation.
    # R[k] for k=0..N-1 is in the first N elements of the real part of irfft(P)
    # Note: The autocorrelation is symmetric.
    R = np.fft.irfft(P, n=n_fft)
    # R[0] is the energy (N for +-1 sequence)
    # R[1] to R[N-1] are the autocorrelations for lags 1 to N-1
    # We want to minimize the max absolute value of these
    # Note: Due to floating point, small imaginary parts might exist, but irfft returns real
    # Also, the linear autocorrelation R_lin[k] = sum f[i]f[i+k] is exactly R[k] for k < N
    # if we zero-pad correctly.
    
    # Extract lags 1 to N-1
    acf_lags = R[1:N]
    max_val = np.max(np.abs(acf_lags))
    return max_val

def _compute_score(f):
    """
    The score is the upper bound on C3.
    Based on inspirations and typical definitions, this is likely 
    max_abs_acf / N.
    """
    max_acf = _compute_max_acf(f)
    return max_acf / N

def _m_sequence(n):
    """
    Generate an M-sequence (Maximum Length Sequence) of length 2^n - 1.
    Since N=400 is not of the form 2^n - 1, we will use a segment or pad.
    Actually, it's better to use a random start or a specific known good sequence.
    For N=400, we can use a Legendre sequence or just random.
    Let's stick to random initialization and SA.
    """
    pass

def _simulated_annealing(f_init, time_budget, rng, T_start=10.0, T_end=0.1, cooling_rate=0.999):
    """
    Simulated Annealing to minimize the max absolute autocorrelation.
    """
    f = f_init.copy()
    current_val = _compute_score(f)
    best_f = f.copy()
    best_val = current_val
    
    T = T_start
    start_time = time.time()
    
    while time.time() < time_budget:
        # Generate a new state by flipping a random bit
        # Flipping one bit changes autocorrelations significantly.
        idx = rng.integers(N)
        f_new = f.copy()
        f_new[idx] *= -1.0
        
        new_val = _compute_score(f_new)
        
        delta = new_val - current_val
        
        if delta < 0:
            f = f_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_f = f.copy()
        else:
            # Accept with probability exp(-delta / T)
            # delta is small (score is ~0.05), T should be scaled appropriately
            # If delta is 0.001 and T is 1.0, prob is exp(-0.001) ~ 0.999
            # We want to accept some worse moves to escape local minima.
            if rng.random() < np.exp(-delta / T):
                f = f_new
                current_val = new_val
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_f, best_val

def _local_search(f_init, time_budget, rng):
    """
    Greedy local search: flip bits if it improves the score.
    """
    f = f_init.copy()
    current_val = _compute_score(f)
    best_f = f.copy()
    best_val = current_val
    start_time = time.time()
    
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        for i in range(N):
            if time.time() > time_budget:
                break
            f[i] *= -1.0
            new_val = _compute_score(f)
            if new_val < current_val - 1e-12:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_f = f.copy()
            else:
                f[i] *= -1.0 # revert
    return best_f, best_val

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialize with random sequence
    f_best = rng.choice([-1.0, 1.0], size=N)
    best_val = _compute_score(f_best)
    
    # Run multiple SA restarts to find a good basin
    # We have ~140s. Let's do 5 restarts of 20s each.
    restarts = 5
    time_per_restart = (time_limit - 10) / restarts # Leave 10s for final local search
    
    for r in range(restarts):
        # Random restart
        f_init = rng.choice([-1.0, 1.0], size=N)
        t_start = time.time()
        t_end = t_start + time_per_restart
        f_candidate, val_candidate = _simulated_annealing(f_init, t_end, rng)
        
        if val_candidate < best_val:
            best_val = val_candidate
            f_best = f_candidate.copy()
            
    # Final polish with local search
    t_start = time.time()
    t_end = t_start + (time_limit - (time.time() - start_time) - 5) # Leave 5s safety
    if t_end > t_start:
        f_best, best_val = _local_search(f_best, t_end, rng)
        
    return f_best
# EVOLVE-BLOCK-END
