"""
Minimize the L-infinity norm of the autocorrelation (excluding lag 0) of a ±1 step function
with N=400 intervals. Uses a fast O(N log N) FFT-based score, multi-start initialization,
and a combination of simulated annealing with block-flip moves and greedy local search.
"""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags using FFT."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(s.astype(np.float64), n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k >= 0
    # For circular autocorrelation we need to be careful, but for linear it's fine.
    # Actually, the FFT gives circular autocorrelation. Let's use linear via padding.
    # With n_fft >= 2N-1, the first N values of irfft give linear autocorrelation.
    return float(np.max(np.abs(acf[1:])))

def _score(s):
    """The score: upper bound on C3. Lower is better."""
    return _autocorr_max_abs(s)

def _initial_seed(rng):
    """
    Create a good initial sign vector using flat-spectrum (Gerchberg-Saxton) optimization.
    """
    s = rng.choice([-1.0, 1.0], size=N)
    
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    for iteration in range(300):
        F = np.fft.rfft(s, n=n_fft)
        magnitude = np.abs(F)
        phase = np.angle(F)
        
        # Flatten magnitude except DC
        mean_mag = np.mean(magnitude[1:])
        new_magnitude = np.full_like(magnitude, mean_mag)
        new_magnitude[0] = magnitude[0]
        
        F_new = new_magnitude * np.exp(1j * phase)
        s_new = np.fft.irfft(F_new, n=n_fft)[:N]
        
        s = np.sign(s_new)
        s[s == 0] = 1.0
    
    return s

def _sa_search(s, time_limit, rng):
    """
    Simulated annealing with both single-bit and block-flip moves.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    current = s.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T_start = 0.3
    T_end = 0.0005
    n_steps = 80000
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        # Choose move type: 70% single flip, 30% block flip
        if rng.random() < 0.7:
            i = rng.integers(0, N)
            current[i] = -current[i]
        else:
            # Flip a random contiguous block of length 2-10
            block_len = rng.integers(2, 11)
            i = rng.integers(0, N - block_len + 1)
            current[i:i+block_len] *= -1.0
        
        new_score = _score(current)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current.copy()
        else:
            delta = new_score - current_score
            if delta > 0 and delta / T < 700:
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    if rng.random() < 0.7:
                        i = rng.integers(0, N)
                        current[i] = -current[i]
                    else:
                        block_len = rng.integers(2, 11)
                        i = rng.integers(0, N - block_len + 1)
                        current[i:i+block_len] *= -1.0
            else:
                # Revert
                if rng.random() < 0.7:
                    i = rng.integers(0, N)
                    current[i] = -current[i]
                else:
                    block_len = rng.integers(2, 11)
                    i = rng.integers(0, N - block_len + 1)
                    current[i:i+block_len] *= -1.0
    
    return best, best_score

def _local_search(s, time_limit):
    """
    Greedy local search: flip individual signs if it improves the score.
    Multiple passes until no improvement or time limit.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    current = s.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            current[i] = -current[i]
            new_score = _score(current)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current.copy()
                improved = True
            else:
                current[i] = -current[i]
    
    return best, best_score

def construct():
    """
    Construct a signed step function to minimize the autoconvolution upper bound.
    Returns an array of N signed step heights (+1 or -1).
    """
    start_time = time.time()
    total_deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Multi-start initialization - generate several seeds, pick the best
    best_signs = None
    best_score = float('inf')
    
    n_starts = 5
    for k in range(n_starts):
        if time.time() > total_deadline - 30:
            break
        seed = _initial_seed(rng)
        sc = _score(seed)
        if sc < best_score:
            best_score = sc
            best_signs = seed.copy()
    
    # Phase 2: Simulated annealing from the best seed
    remaining = total_deadline - time.time()
    if remaining > 10.0:
        sa_time = min(remaining - 5.0, 60.0)
        if sa_time > 5.0:
            sa_result, sa_score = _sa_search(best_signs, sa_time, rng)
            if sa_score < best_score:
                best_signs = sa_result
                best_score = sa_score
    
    # Phase 3: Local search
    remaining = total_deadline - time.time()
    if remaining > 5.0:
        ls_time = min(remaining - 3.0, 40.0)
        ls_result, ls_score = _local_search(best_signs, ls_time)
        if ls_score < best_score:
            best_signs = ls_result
            best_score = ls_score
    
    # Phase 4: Another SA round if time remains
    remaining = total_deadline - time.time()
    if remaining > 10.0:
        sa_time2 = min(remaining - 5.0, 30.0)
        if sa_time2 > 5.0:
            sa_result2, sa_score2 = _sa_search(best_signs, sa_time2, rng)
            if sa_score2 < best_score:
                best_signs = sa_result2
                best_score = sa_score2
    
    # Final local search pass
    remaining = total_deadline - time.time()
    if remaining > 3.0:
        ls_result, ls_score = _local_search(best_signs, remaining - 2.0)
        if ls_score < best_score:
            best_signs = ls_result
            best_score = ls_score
    
    return best_signs

# EVOLVE-BLOCK-END
