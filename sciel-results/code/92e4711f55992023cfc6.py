"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autocorr_fast(s):
    """Compute autocorrelation using FFT for lags 0..N-1."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf


def _peak_sidelobe(ac):
    """Peak of |ac[k]| for k=1..N-1."""
    if len(ac) <= 1:
        return 0.0
    return float(np.max(np.abs(ac[1:])))


def _greedy_flip(s, ac, deadline):
    """
    Greedy local search: try flipping each bit, accept if it reduces peak sidelobe.
    Uses O(N) delta computation per bit flip (no FFT).
    Returns updated s, ac, and peak value.
    """
    n = N
    best_val = _peak_sidelobe(ac)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return s, ac, best_val
            old = s[i]
            max_k = min(i, n - 1 - i)
            if max_k < 0:
                continue
            
            # Compute delta for each affected lag k in 1..max_k
            # R(k) = sum_{j=0}^{N-1-k} s[j]*s[j+k]
            # s[i] appears in R(k) as s[i]*s[i+k] (if i+k < N) and s[i-k]*s[i] (if i-k >= 0)
            # Flipping s[i] from old to -old:
            #   delta_R(k) = -2*old*s[i+k] - 2*old*s[i-k]  for k where both terms exist
            #   For k=0: R(0) unchanged since s[i]^2=1 always
            
            new_ac = ac.copy()
            for k in range(1, max_k + 1):
                delta = -2.0 * old * s[i + k] - 2.0 * old * s[i - k]
                new_ac[k] = ac[k] + delta
            
            new_val = _peak_sidelobe(new_ac)
            if new_val < best_val:
                s[i] = -old
                ac = new_ac
                best_val = new_val
                improved = True
    
    return s, ac, best_val


def _simulated_annealing(s, ac, deadline):
    """
    Simulated annealing with O(N) delta updates.
    """
    n = N
    current_val = _peak_sidelobe(ac)
    best_val = current_val
    best_s = s.copy()
    best_ac = ac.copy()
    
    T = 2.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < 1e-6:
            break
        i = np.random.randint(n)
        old = s[i]
        max_k = min(i, n - 1 - i)
        
        if max_k >= 1:
            new_ac = ac.copy()
            for k in range(1, max_k + 1):
                delta = -2.0 * old * s[i + k] - 2.0 * old * s[i - k]
                new_ac[k] = ac[k] + delta
            new_val = _peak_sidelobe(new_ac)
        else:
            new_val = current_val  # No change for k>=1
        
        delta_val = new_val - current_val
        if delta_val < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta_val / T)):
            s[i] = -old
            ac = new_ac
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
                best_ac = ac.copy()
        
        T *= cooling
    
    return best_s, best_ac, best_val


def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    # Initialize with Thue-Morse sequence for good starting point
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map to ±1: +1 if t(n) even, -1 if odd
    s = np.ones(N, dtype=np.float64)
    for i in range(N):
        # Count bits
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        if cnt % 2 == 1:
            s[i] = -1.0
    
    ac = _compute_autocorr_fast(s)
    
    # Phase 1: Greedy local search
    s, ac, val = _greedy_flip(s, ac, deadline)
    
    if time.time() > deadline:
        return s
    
    # Phase 2: Simulated annealing to escape local minima
    s, ac, val = _simulated_annealing(s, ac, deadline)
    
    if time.time() > deadline:
        return s
    
    # Phase 3: Final greedy polish
    s, ac, val = _greedy_flip(s, ac, deadline)
    
    return s
# EVOLVE-BLOCK-END
