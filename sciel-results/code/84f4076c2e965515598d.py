import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    seq = [1]
    while len(seq) < n:
        seq = seq + [-x for x in seq]
    return np.array(seq[:n], dtype=float)

def _autoconv_peak_fft(s):
    """Compute max |autoconv| using FFT."""
    n = len(s)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    S = np.fft.rfft(s, size)
    ac = np.fft.irfft(S * np.conj(S), size)[:2 * n - 1]
    return np.max(np.abs(ac))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = np.inf
    best_s = None
    
    # Start with Thue-Morse
    s = _thue_morse(N)
    current_score = _autoconv_peak_fft(s)
    best_score = current_score
    best_s = s.copy()
    
    rng = np.random.default_rng(123)
    
    # Local search: try flipping each position, keep if it improves
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            s[i] = -s[i]
            new_score = _autoconv_peak_fft(s)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_s = s.copy()
                improved = True
            else:
                s[i] = -s[i]  # revert
    
    # Try some random restarts if time permits
    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline:
                break
            s = rng.choice([-1.0, 1.0], size=N)
            current_score = _autoconv_peak_fft(s)
            # Quick local search
            for _ in range(200):
                if time.time() > deadline:
                    break
                i = rng.integers(0, N)
                s[i] = -s[i]
                new_score = _autoconv_peak_fft(s)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_s = s.copy()
                else:
                    s[i] = -s[i]
    
    if best_s is None:
        return np.ones(N)
    return best_s
