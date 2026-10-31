import numpy as np
import time

N = 400
DEADLINE = 130

def _max_sidelobe(f):
    """Max |autocorrelation| for lags 1..N-1."""
    # Use FFT for efficiency
    n = len(f)
    nfft = 1
    while nfft < 2*n:
        nfft *= 2
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    return np.max(np.abs(acf[1:]))

def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)
    
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _max_sidelobe(best_f) / N
    
    # Local search with restarts
    max_iter = 50
    for restart in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
        
        f = best_f.copy()
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            # Random order of positions
            order = rng.permutation(N)
            for i in order:
                if time.time() - start_time > DEADLINE:
                    break
                f[i] *= -1
                score = _max_sidelobe(f) / N
                if score < best_score - 1e-12:
                    best_score = score
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] *= -1
    
    return best_f
