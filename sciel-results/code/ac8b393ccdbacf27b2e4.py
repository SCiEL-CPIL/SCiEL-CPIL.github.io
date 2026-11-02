"""Rewritten program: minimize L-infinity norm of autocorrelation (excluding lag 0)
using simulated annealing with a fast FFT-based score and a good initial seed."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(s.astype(np.float64), n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] is sum s[i]s[i+k]
    # We exclude lag 0
    return float(np.max(np.abs(acf[1:])))

def _score(s):
    """The score: upper bound on C3. Lower is better."""
    return _autocorr_max_abs(s)

def _initial_seed():
    """
    Create a good initial sign vector using a gradient-like approach on
    a continuous relaxation, then binarize.
    
    We minimize the L2 norm of the autocorrelation (flat spectrum) via
    iterative projection, which gives a low-peak-sidelobe starting point.
    """
    # Start with a random ±1 vector
    rng = np.random.default_rng(42)
    s = rng.choice([-1.0, 1.0], size=N)
    
    # Iterative "flat spectrum" optimization:
    # Alternate between time-domain (keep ±1) and frequency-domain (flatten magnitude)
    # This is similar to the Gerchberg-Saxton / Fienup algorithm.
    
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    for iteration in range(200):
        # Forward FFT
        F = np.fft.rfft(s, n=n_fft)
        
        # In frequency domain: try to make |F| as flat as possible (except DC)
        # Replace |F| with its mean (for non-DC bins), keep phase
        magnitude = np.abs(F)
        phase = np.angle(F)
        
        # Set magnitude to mean of non-DC magnitudes
        mean_mag = np.mean(magnitude[1:])
        new_magnitude = np.full_like(magnitude, mean_mag)
        new_magnitude[0] = magnitude[0]  # Keep DC
        
        # Inverse FFT with new magnitudes
        F_new = new_magnitude * np.exp(1j * phase)
        s_new = np.fft.irfft(F_new, n=n_fft)[:N]
        
        # Project back to ±1
        s = np.sign(s_new)
        s[s == 0] = 1.0
    
    return s

def _sa_search(s, time_limit):
    """
    Simulated annealing to minimize the L-infinity autocorrelation score.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    current = s.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Temperature schedule
    T_start = 0.5
    T_end = 0.001
    n_steps = 50000
    
    # Precompute indices for random flips
    rng = np.random.default_rng(123)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Linear cooling schedule
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        # Pick a random index to flip
        i = rng.integers(0, N)
        
        old_sign = current[i]
        current[i] = -old_sign
        
        new_score = _score(current)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            delta = new_score - current_score
            if delta > 0 and delta / T < 700:  # Avoid overflow
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current[i] = old_sign
            else:
                current[i] = old_sign
    
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
            old_sign = current[i]
            current[i] = -old_sign
            new_score = _score(current)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current.copy()
                improved = True
            else:
                current[i] = old_sign
    
    return best, best_score

def construct():
    """
    Construct a signed step function to minimize the autoconvolution upper bound.
    Returns an array of N signed step heights (+1 or -1).
    """
    start_time = time.time()
    total_deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: Generate a good initial seed via flat-spectrum optimization
    seed = _initial_seed()
    seed_score = _score(seed)
    
    best_signs = seed.copy()
    best_score = seed_score
    
    # Phase 2: Simulated annealing
    sa_time = min(60.0, 0.5 * (140.0 - (time.time() - start_time)))
    if sa_time > 5.0:
        sa_result, sa_score = _sa_search(seed, sa_time)
        if sa_score < best_score:
            best_signs = sa_result
            best_score = sa_score
    
    # Phase 3: Local search from the best so far
    remaining = 140.0 - (time.time() - start_time)
    if remaining > 5.0:
        ls_time = min(remaining - 5.0, 50.0)
        ls_result, ls_score = _local_search(best_signs, ls_time)
        if ls_score < best_score:
            best_signs = ls_result
            best_score = ls_score
    
    # Phase 4: Another SA round from the locally-optimal point if time remains
    remaining = 140.0 - (time.time() - start_time)
    if remaining > 10.0:
        sa_time2 = min(remaining - 5.0, 30.0)
        if sa_time2 > 5.0:
            sa_result2, sa_score2 = _sa_search(best_signs, sa_time2)
            if sa_score2 < best_score:
                best_signs = sa_result2
                best_score = sa_score2
    
    # Final local search pass
    remaining = 140.0 - (time.time() - start_time)
    if remaining > 3.0:
        ls_result, ls_score = _local_search(best_signs, remaining - 2.0)
        if ls_score < best_score:
            best_signs = ls_result
            best_score = ls_score
    
    return best_signs

# EVOLVE-BLOCK-END
