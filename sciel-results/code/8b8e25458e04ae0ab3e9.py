"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _linear_autocorr(s):
    """Compute linear autocorrelation of a length-N sequence.
    Returns array of length 2N-1 where index k corresponds to lag k-N.
    Specifically, acorr[lag + N] = sum_i s[i]*s[i+lag] for lag>=0,
    and acorr[N-lag] = sum_i s[i]*s[i-lag] for lag<0 (symmetric).
    """
    # Use FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(s, n=n_fft)
    acorr_full = np.fft.irfft(f * np.conj(f), n=n_fft)
    # The linear autocorrelation is in the first N and last N-1 elements
    acorr = np.concatenate([acorr_full[N-1:], acorr_full[:N]])
    return acorr


def _c3_bound(s):
    """Compute the C3 upper bound for the given step function heights.
    
    The third autocorrelation inequality involves bounding the integral of
    |f|^3 in terms of autocorrelation quantities. For a step function with
    heights s[i] on N equal intervals, the relevant quantity is:
    
    C3_bound = max over lags of |sum_i s[i] s[i+lag]|^2 / (sum s[i]^2)^2 * sum s[i]^2
              ... actually let me think about this more carefully.
    
    The task says "improve an autoconvolution upper bound" and "score is the
    upper bound on C3". The seed returns np.ones(N) and scores 1.0.
    
    For the standard autocorrelation inequality context: if we have a function
    f on [-1/4, 1/4] with ||f||_1 = 1 (or some normalization), the C3 constant
    relates to sup over shifts of |f * f~(shift)| where f~ is the reflection.
    
    For a step function with heights s[i], the autoconvolution at shift k
    (in units of interval width) is sum_i s[i] s[i+k]. The L-infinity norm
    of the autoconvolution is max_k |sum_i s[i] s[i+k]|.
    
    But the "upper bound on C3" likely refers to a specific inequality. Given
    the seed is all-ones with score 1.0, and the task mentions "signed step
    function", let me reconsider.
    
    Actually, re-reading: "Choose a signed step function with 400 equal
    intervals on [-1/4,1/4] to improve an autoconvolution upper bound."
    
    The C3 constant in harmonic analysis often appears in inequalities like:
    ||f * f||_infty <= C3 * ||f||_2^2
    
    For a step function with heights s[i] (each interval has height s[i]),
    if we normalize so that sum |s[i]| = 1 (L1 norm = 1 over the interval),
    then ||f||_2^2 = sum s[i]^2 * (interval_width) and 
    (f*f)(k) = sum_i s[i] s[i+k] * (interval_width).
    
    The ratio ||f*f||_infty / ||f||_2^2 = max_k |sum_i s[i]s[i+k]| / sum_i s[i]^2
    
    For all-ones with L1=1: s[i] = 1/400, sum s[i]^2 = 400*(1/400)^2 = 1/400
    max autocorr = sum_i (1/400)^2 * 400 = 400/160000 = 1/400
    Ratio = (1/400)/(1/400) = 1.0 ✓
    
    So the score is: max_k |sum_i s[i] s[i+k]| / sum_i s[i]^2
    where s is the sequence of heights, and we want to MINIMIZE this.
    
    Wait, but "signed" means s[i] can be negative. And we want to minimize
    the upper bound. Let me verify: for all +1's, the autocorrelation at lag 0
    is sum s[i]^2, and at other lags it's also positive. The max is at lag 0
    (or near it). The ratio is 1.
    
    If we use alternating +1, -1, +1, -1, ..., then autocorrelation at even
    lags is positive, at odd lags is 0 (approximately). The max would be at
    lag 0: sum s[i]^2 = 400 (if s[i] in {+1,-1}). sum s[i]^2 = 400.
    Ratio = 400/400 = 1. Same!
    
    Hmm. Let me reconsider. Maybe the normalization is different.
    
    Actually, maybe it's: the function f has values s[i] on each interval,
    and we consider the autoconvolution (f * f_reflected)(t) and bound it
    by C3 * ||f||_2^2. The constant C3 for this specific f is
    sup_t |f*f_ref(t)| / ||f||_2^2.
    
    For discrete step functions with N intervals of width w = (1/2)/N = 1/(2N):
    ||f||_2^2 = w * sum s[i]^2
    (f * f_ref)(k*w) = w * sum_i s[i] s[i+k]  (for appropriate lag k)
    
    So C3(f) = max_k |w * sum_i s[i]s[i+k]| / (w * sum s[i]^2)
             = max_k |sum_i s[i]s[i+k]| / sum s[i]^2
    
    This is what I had. For all +1: max autocorr = N (at lag 0), sum s^2 = N, ratio = 1.
    For alternating: autocorr at lag 0 = N, at lag 1 = 0, at lag 2 = N, etc.
    Max is N, ratio = N/N = 1.
    
    To get below 1, we need the max absolute autocorrelation to be less than
    sum of squares. This is like designing a sequence with low peak
    autocorrelation relative to its energy - similar to Barker sequences or
    Zadoff-Chu, but we can use real-valued (signed) heights.
    
    The theoretical minimum for real sequences: by Parseval, the sum of
    squared autocorrelations equals (sum s[i]^2)^2. So if there are 2N-1
    lags, the minimum possible max |autocorr| is at least
    (sum s^2)^2 / (2N-1) in RMS sense, but we want the peak.
    
    Actually, sum_k |autocorr(k)|^2 = (sum_i s[i]^2)^2. So the average
    squared autocorrelation is (sum s^2)^2 / (2N-1). The max is at least
    the sqrt of the average, so max |autocorr| >= (sum s^2) / sqrt(2N-1).
    Thus C3 >= 1/sqrt(2N-1) = 1/sqrt(799) ≈ 0.0354.
    
    So there's a lot of room to improve from 1.0 down to ~0.035.
    """
    s = np.asarray(s, dtype=np.float64)
    energy = np.dot(s, s)
    if energy < 1e-15:
        return 1.0  # Degenerate
    acorr = _linear_autocorr(s)
    peak = np.max(np.abs(acorr))
    return peak / energy


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n with values +1/-1."""
    seq = np.zeros(n)
    seq[0] = 1
    for i in range(1, n):
        # Count trailing 1s in binary representation of i
        t = i
        parity = 0
        while t:
            parity ^= (t & 1)
            t >>= 1
        seq[i] = 1 if parity == 0 else -1
    return seq


def _m_sequence(n):
    """Generate an m-sequence-like sequence. For non-power-of-2, use
    a simple periodic construction or fall back to Thue-Morse."""
    # Use a simple maximal-length LFSR if n is close to 2^m - 1
    # Otherwise, use a different construction
    if n == 2**7 - 1:  # 127
        # Standard 7-bit LFSR
        pass
    # For general n, use a sine-based or other construction
    # Let's use a pseudo-random sequence with good autocorrelation properties
    # Actually, for flexibility, let's just use Thue-Morse or random
    return _thue_morse(n)


def construct():
    """Construct a signed step function with N=400 intervals to minimize
    the C3 autoconvolution upper bound."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Initialize with several diverse starting points
    candidates = []
    
    # 1. Thue-Morse
    tm = _thue_morse(N)
    candidates.append(tm.copy())
    
    # 2. Random signed
    np.random.seed(42)
    rand_seq = np.random.choice([-1.0, 1.0], size=N)
    candidates.append(rand_seq)
    
    # 3. Sine-modulated
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    sine_seq = np.sign(np.sin(x))
    candidates.append(sine_seq)
    
    # 4. Another random
    np.random.seed(123)
    rand_seq2 = np.random.choice([-1.0, 1.0], size=N)
    candidates.append(rand_seq2)
    
    # 5. Alternating blocks
    block_seq = np.tile([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], N // 8)
    block_seq = block_seq[:N]
    candidates.append(block_seq)
    
    # Evaluate initial candidates
    best_seq = None
    best_score = float('inf')
    for s in candidates:
        score = _c3_bound(s)
        if score < best_score:
            best_score = score
            best_seq = s.copy()
    
    # Simulated Annealing with hybrid moves
    if best_seq is None:
        best_seq = np.ones(N)
        best_score = 1.0
    
    current_seq = best_seq.copy()
    current_score = best_score
    
    # SA parameters
    T_start = 0.5
    T_end = 0.001
    cooling = 0.999
    T = T_start
    
    np.random.seed(2024)
    
    while time.time() < deadline:
        # Choose move type
        move_type = np.random.randint(3)
        
        if move_type == 0:
            # Flip a single element
            idx = np.random.randint(N)
            new_seq = current_seq.copy()
            new_seq[idx] = -new_seq[idx]
        elif move_type == 1:
            # Swap two elements
            i, j = np.random.choice(N, size=2, replace=False)
            new_seq = current_seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        else:
            # Adjacent swap
            i = np.random.randint(N - 1)
            j = i + 1
            new_seq = current_seq.copy()
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_score = _c3_bound(new_seq)
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_seq = new_seq
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = current_seq.copy()
        
        T *= cooling
        if T < T_end:
            T = T_end
    
    # Final polish: local optimization on the best sequence
    # Try flipping each element and keeping improvements
    for _ in range(5):  # Multiple passes
        if time.time() > deadline - 5:
            break
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                old_val = best_seq[i]
                best_seq[i] = -old_val
                test_score = _c3_bound(best_seq)
                if test_score < best_score:
                    best_score = test_score
                    current_seq = best_seq.copy()
                    improved = True
                else:
                    best_seq[i] = old_val
    
    return best_seq
# EVOLVE-BLOCK-END
