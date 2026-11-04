"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak_sidelobe(h):
    """Compute max absolute autocorrelation at non-zero lags (peak sidelobe)."""
    n = len(h)
    if n <= 1:
        return 0.0
    # FFT-based autocorrelation
    fft_size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(H * np.conj(H), n=fft_size)
    # acf[0] is the DC term (sum of squares), acf[1:n] are positive lags
    # For real signals, negative lags are symmetric
    return float(np.max(np.abs(acf[1:n])))


def _thue_morse(n):
    """Generate Thue-Morse sequence mapped to ±1."""
    seq = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        # Parity of popcount
        if (i & 1) ^ ((i >> 1) & 1) ^ ((i >> 2) & 1) ^ ((i >> 3) & 1) ^ \
           ((i >> 4) & 1) ^ ((i >> 5) & 1) ^ ((i >> 6) & 1) ^ ((i >> 7) & 1):
            seq[i] = -1.0
    return seq


def _chirp_init(n, alpha=1.0, beta=0.5):
    """Generate sign of sum of linear and quadratic chirp phases."""
    t = np.arange(n, dtype=np.float64)
    phase_linear = alpha * t
    phase_quadratic = beta * t * t
    # Sum of complex exponentials, take sign of real part
    signal = np.cos(phase_linear) + np.cos(phase_quadratic)
    return np.sign(signal)


def _fast_local_search(h, time_limit=100.0, rng_seed=42):
    """Fast local search: try flipping individual signs, keep improvements."""
    rng = np.random.RandomState(rng_seed)
    best = h.copy().astype(np.float64)
    best_score = _autocorr_peak_sidelobe(best)
    start = time.time()
    n = len(best)
    
    # Precompute for speed: use the fact that flipping h[i] changes autocorrelation
    # by a known amount, but for simplicity just recompute with FFT
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Random order for diversity
        order = rng.permutation(n)
        for i in order:
            if time.time() - start > time_limit - 5.0:
                break
            trial = best.copy()
            trial[i] = -trial[i]
            s = _autocorr_peak_sidelobe(trial)
            if s < best_score - 1e-12:
                best = trial
                best_score = s
                improved = True
    
    return best, best_score


def _multi_start_search(time_limit=100.0):
    """Try multiple initializations and local search from each."""
    start = time.time()
    candidates = []
    
    # Thue-Morse
    tm = _thue_morse(N)
    candidates.append(tm)
    
    # Chirp-based with different parameters
    for alpha, beta in [(1.0, 0.5), (2.0, 1.0), (0.5, 2.0), (1.0, 1.0)]:
        candidates.append(_chirp_init(N, alpha, beta))
    
    # Random
    rng = np.random.RandomState(0)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    best_h = None
    best_score = float('inf')
    
    for c in candidates:
        if time.time() - start > time_limit - 10.0:
            break
        # Quick local search from this candidate
        h, score = _fast_local_search(c, time_limit=min(15.0, time_limit - (time.time() - start) - 5.0), rng_seed=hash(c.tobytes()) % 2**31)
        if score < best_score:
            best_score = score
            best_h = h
    
    if best_h is None:
        best_h = tm
        best_score = _autocorr_peak_sidelobe(tm)
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Multi-start with local search
    h, score = _multi_start_search(time_limit=min(120.0, deadline - time.time() - 10.0))
    
    # Final check
    if time.time() > deadline:
        # Fallback to Thue-Morse
        h = _thue_morse(N)
    
    return h
# EVOLVE-BLOCK-END
