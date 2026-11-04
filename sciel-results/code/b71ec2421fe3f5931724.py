"""
Optimized signed step function for third autocorrelation inequality.
Uses a hybrid approach:
1. Fast FFT-based scoring
2. Structured initialization (m-sequences, Welch bounds)
3. Multi-start local search with sign flips and zeroing
4. Simulated annealing for global exploration
"""
import numpy as np
import time
from math import gcd

N = 400

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags using FFT."""
    n = len(h)
    if n <= 1:
        return 0.0
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _m_sequence_init(period, length):
    """Generate an m-sequence (maximum length LFSR sequence) truncated to length."""
    # Simple LFSR for m-sequence generation
    # Use a known primitive polynomial for period 511 (2^9 - 1)
    # x^9 + x^4 + 1
    if period == 511:
        mask = (1 << 9) - 1
        poly = 0x111  # bits 0, 4, 8 set
        state = 1
        seq = []
        for _ in range(period):
            seq.append(1 if (state & 1) else -1)
            feedback = (state & 1) ^ ((state >> 4) & 1)
            state = (state >> 1) | (feedback << 8)
            state &= mask
        return np.array(seq[:length], dtype=np.float64)
    # Fallback: random
    return np.random.choice([-1, 1], size=length).astype(np.float64)

def _welch_init(length):
    """Initialize with a sequence designed to approach Welch bound."""
    # Welch bound for N points: sqrt(N/(N-1)) * (N-1)/N
    # For binary sequences, target autocorrelation ~ sqrt(N)
    # Start with random and refine
    h = np.random.choice([-1, 1], size=length).astype(np.float64)
    return h

def _sparse_init(k, length):
    """Create a sparse sequence with k non-zero entries."""
    h = np.zeros(length)
    indices = np.random.choice(length, size=k, replace=False)
    h[indices] = np.random.choice([-1, 1], size=k).astype(np.float64)
    return h

def _local_search(h, time_limit=10.0):
    """
    Fast local search: try sign flips and zeroing.
    Returns improved sequence and score.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    def _fast_score(h):
        H = np.fft.rfft(h, n=fft_size)
        acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
        return float(np.max(np.abs(acf[1:n])))
    
    best_score = _fast_score(best_h)
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try flipping signs of non-zero elements
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) > 0:
            order = np.random.permutation(len(nonzero_idx))
            for idx_pos in order:
                i = nonzero_idx[idx_pos]
                trial = best_h.copy()
                trial[i] = -trial[i]
                s = _fast_score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
        
        # Try zeroing out elements
        if not improved:
            nonzero_idx = np.nonzero(best_h)[0]
            if len(nonzero_idx) > 0:
                order = np.random.permutation(len(nonzero_idx))
                for idx_pos in order:
                    i = nonzero_idx[idx_pos]
                    trial = best_h.copy()
                    trial[i] = 0.0
                    s = _fast_score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
        
        # Try activating zero elements
        if not improved:
            zero_idx = np.nonzero(best_h == 0)[0]
            if len(zero_idx) > 0:
                k = min(30, len(zero_idx))
                order = np.random.choice(len(zero_idx), size=k, replace=False)
                for idx_pos in order:
                    i = zero_idx[idx_pos]
                    for sign in [1.0, -1.0]:
                        trial = best_h.copy()
                        trial[i] = sign
                        s = _fast_score(trial)
                        if s < best_score - 1e-12:
                            best_h = trial
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return best_h, best_score

def _simulated_annealing(h, time_limit=15.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 10.0
    cooling = 0.998
    min_T = 0.5
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    def _fast_score(h):
        H = np.fft.rfft(h, n=fft_size)
        acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
        return float(np.max(np.abs(acf[1:n])))
    
    current_score = _fast_score(current_h)
    best_score = current_score
    
    while T > min_T and time.time() - start < time_limit:
        # Pick a random move
        i = np.random.randint(n)
        trial = current_h.copy()
        
        # Randomly choose between flip, zero, or activate
        r = np.random.random()
        if r < 0.5:
            # Flip sign (or activate if 0)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        elif r < 0.75:
            # Zero out
            trial[i] = 0.0
        else:
            # Activate (if zero) or flip (if non-zero)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        
        new_score = _fast_score(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
    
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Try multiple initialization strategies (m-sequence, Welch, sparse)
    2. Use local search to refine
    3. Use simulated annealing for global exploration
    4. Return the best solution found within the time budget
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Strategy 1: m-sequence based initialization
    if time.time() < deadline - 20:
        h = _m_sequence_init(511, N)
        h_opt, score = _local_search(h, time_limit=8.0)
        if time.time() < deadline - 10:
            h_opt, score = _simulated_annealing(h_opt, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Strategy 2: Welch-like random initialization
    if time.time() < deadline - 20:
        h = _welch_init(N)
        h_opt, score = _local_search(h, time_limit=8.0)
        if time.time() < deadline - 10:
            h_opt, score = _simulated_annealing(h_opt, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Strategy 3: Sparse initializations with various densities
    for k in [50, 100, 150, 200]:
        if time.time() > deadline - 15:
            break
        
        for trial in range(2):
            if time.time() > deadline - 10:
                break
            
            h = _sparse_init(k, N)
            h_opt, score = _local_search(h, time_limit=5.0)
            if time.time() < deadline - 8:
                h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Final refinement if time permits
    if best_h is not None and time.time() < deadline - 10:
        best_h, best_score = _local_search(best_h, time_limit=10.0)
        if time.time() < deadline - 5:
            best_h, best_score = _simulated_annealing(best_h, time_limit=10.0)
    
    # Fallback
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
