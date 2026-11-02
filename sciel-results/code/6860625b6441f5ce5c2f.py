"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute FFT size
_FFT_SIZE = 1
while _FTT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _compute_acf(h):
    """Compute aperiodic autocorrelation using FFT."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    ac = _compute_acf(h)
    return float(np.max(np.abs(ac[1:])))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n mapped to +/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map 0 -> 1, 1 -> -1
    seq = np.zeros(n)
    for i in range(n):
        # Count set bits
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1.0 if (count % 2 == 0) else -1.0
    return seq

def _anneal_fast(seed, time_budget, init_h=None):
    """
    Simulated annealing with fast analytic single-bit updates.
    Instead of recomputing full ACF via FFT for each step, we update
    the ACF analytically when flipping a single bit.
    
    However, for simplicity and robustness, we use a hybrid approach:
    - Use FFT-based evaluation but with a tighter inner loop
    - Add block flips for exploration
    """
    rng = np.random.default_rng(seed)
    
    if init_h is not None:
        h = init_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-7
    start = time.time()
    
    while time.time() - start < time_budget:
        # Variable block length
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(16.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99995)
    
    return best_h, best

def _greedy_polish_single(h, time_budget, start):
    """Greedy single-flip polish."""
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best

def _greedy_polish_pair(h, time_budget, start):
    """Greedy adjacent-pair-flip polish."""
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_pair = -1
        best_c = cur
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
        if best_pair < 0:
            break
        h[best_pair] *= -1.0
        h[best_pair + 1] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best

def construct():
    try:
        hard_deadline = 145.0  # Safety margin from 150s limit
        start = time.time()
        
        # Initialize with Thue-Morse sequence
        global_best_h = _thue_morse(N)
        global_best = _peak_sidelobe(global_best_h)
        
        # First anneal run from Thue-Morse
        budget = min(20.0, hard_deadline - (time.time() - start) - 5.0)
        if budget > 1.0:
            h, c = _anneal_fast(12345, budget, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Multiple shorter restarts for diversity
        restart_id = 1
        while time.time() - start < hard_deadline - 10.0:
            sub = min(8.0, hard_deadline - (time.time() - start) - 10.0)
            if sub < 1.0:
                break
            h, c = _anneal_fast(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1
        
        # Final greedy polish: single-flip
        remaining = hard_deadline - (time.time() - start) - 3.0
        if remaining > 1.0:
            h, c = _greedy_polish_single(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Pair-flip polish if time remains
        remaining = hard_deadline - (time.time() - start) - 1.0
        if remaining > 0.5:
            h, c = _greedy_polish_pair(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
        
        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)

# EVOLVE-BLOCK-END
