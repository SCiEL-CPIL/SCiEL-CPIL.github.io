"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _autocorr_peak(s):
    """Compute the maximum absolute value of aperiodic autocorrelation at non-zero lags."""
    n = len(s)
    # Use FFT for efficient autocorrelation
    size = 1
    while size < 2 * n:
        size *= 2
    s_pad = np.zeros(size, dtype=np.float64)
    s_pad[:n] = s
    S = np.fft.rfft(s_pad, size)
    acf_full = np.fft.irfft(S * np.conj(S), size)
    # Aperiodic autocorrelation: acf[k] = sum_{i=0}^{n-1-k} s[i]*s[i+k] for k>=0
    # The FFT gives circular autocorrelation, but with zero-padding it gives aperiodic
    # For lag k (0-indexed), the aperiodic autocorrelation is at index k
    acf = acf_full[:n]
    if len(acf) <= 1:
        return 0.0
    return np.max(np.abs(acf[1:]))


def _score(s):
    """The score is the upper bound on C3, which we take as the peak sidelobe of autocorrelation."""
    return _autocorr_peak(s)


def _fast_greedy(n, seed=0):
    """
    Greedy construction of a ±1 sequence with low autocorrelation peak.
    At each step, choose the sign that minimizes the current peak sidelobe.
    """
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[0] = 1.0
    for i in range(1, n):
        # Try both signs
        best_val = float('inf')
        best_sign = 1.0
        for sign in [1.0, -1.0]:
            s[i] = sign
            # Compute peak sidelobe for the prefix
            prefix = s[:i+1]
            # Quick estimate: just check the new lag contributions
            # For efficiency, we can approximate by checking only recent lags
            # But for correctness, let's do a partial check
            # Actually, let's just compute the full peak for the prefix
            # This is O(n^2) total, which for n=400 is 160000 ops, acceptable
            peak = _autocorr_peak(prefix)
            if peak < best_val:
                best_val = peak
                best_sign = sign
        s[i] = best_sign
    return s


def _legendre_init(n, seed=0):
    """
    Initialize using a Legendre sequence-like approach.
    Legendre sequences have good autocorrelation properties.
    """
    rng = np.random.default_rng(seed)
    # Start with a random ±1 sequence and apply local improvements
    s = rng.choice([-1.0, 1.0], size=n)
    return s


def _quasiperiodic_init(n, seed=0):
    """
    Initialize using a quasiperiodic (irrational rotation) sequence.
    This can produce sequences with good spectral properties.
    """
    rng = np.random.default_rng(seed)
    # Use golden ratio as an irrational number for quasiperiodic sequence
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    alpha = 1.0 / phi  # ~0.618
    # Random offset
    offset = rng.uniform(0, 1)
    s = np.zeros(n)
    for i in range(n):
        x = (i * alpha + offset) % 1.0
        s[i] = 1.0 if x < 0.5 else -1.0
    return s


def _simulated_annealing(s_init, time_limit, seed=0):
    """
    Simulated annealing on a ±1 sequence to minimize the peak sidelobe.
    """
    rng = np.random.default_rng(seed)
    s = s_init.copy()
    current_score = _autocorr_peak(s)
    best_score = current_score
    best_s = s.copy()
    
    start_time = time.time()
    T = 1.0  # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        if T < 1e-6:
            break
            
        # Random move: flip a random position
        i = rng.integers(0, len(s))
        old_val = s[i]
        s[i] = -old_val
        
        new_score = _autocorr_peak(s)
        delta = new_score - current_score
        
        # Accept with probability exp(-delta/T) if delta > 0
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = s.copy()
        else:
            s[i] = old_val
        
        T *= cooling_rate
    
    return best_s, best_score


def construct():
    """
    Construct a signed step function with N=400 intervals on [-1/4, 1/4]
    that minimizes the autoconvolution upper bound (C3).
    Uses dense ±1 sequences with simulated annealing and multiple initialization strategies.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_overall_score = float('inf')
    best_overall_vec = np.zeros(N)
    
    # Strategy 1: Greedy construction
    if time.time() < deadline - 30:
        greedy_s = _fast_greedy(N, seed=42)
        greedy_score = _autocorr_peak(greedy_s)
        if greedy_score < best_overall_score:
            best_overall_score = greedy_score
            best_overall_vec = greedy_s.copy()
    
    # Strategy 2: Multiple simulated annealing runs with different initializations
    init_strategies = [
        ('legendre', lambda seed: _legendre_init(N, seed)),
        ('quasiperiodic', lambda seed: _quasiperiodic_init(N, seed)),
        ('random', lambda seed: np.random.default_rng(seed).choice([-1.0, 1.0], size=N)),
    ]
    
    num_sa_runs = 5
    sa_time_each = 15.0  # seconds per SA run
    
    for run_idx in range(num_sa_runs):
        if time.time() > deadline - 5:
            break
        
        # Choose initialization strategy
        strat_name, strat_func = init_strategies[run_idx % len(init_strategies)]
        seed = 1000 + run_idx * 137
        s_init = strat_func(seed)
        
        # Simulated annealing
        sa_time = min(sa_time_each, deadline - time.time() - 3)
        if sa_time < 1.0:
            break
        
        sa_s, sa_score = _simulated_annealing(s_init, sa_time, seed=seed)
        
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall_vec = sa_s.copy()
    
    # Strategy 3: If time remains, do a final refinement on the best solution
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5.0 and best_overall_score < float('inf'):
        refined_s, refined_score = _simulated_annealing(
            best_overall_vec, remaining_time, seed=9999
        )
        if refined_score < best_overall_score:
            best_overall_score = refined_score
            best_overall_vec = refined_s.copy()
    
    # Ensure the result is a valid ±1 sequence
    best_overall_vec = np.sign(best_overall_vec)
    best_overall_vec[best_overall_vec == 0] = 1.0
    
    return best_overall_vec
# EVOLVE-BLOCK-END
