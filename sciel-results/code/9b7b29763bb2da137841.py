"""Rewritten program: minimize max|autocorr| / autocorr[0] via efficient random search."""
import numpy as np
import time

N = 400

def _score(x):
    """Compute max|autocorr| / autocorr[0]. Returns inf if autocorr[0] ~ 0."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # Full autocorrelation via FFT
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    ac0 = ac[0]
    if abs(ac0) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(ac)) / ac0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with all-ones (score 1.0)
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Phase 1: Random search
    # We perform a fixed number of high-quality random restarts.
    # Each restart involves generating a signal and optionally a brief local search.
    
    # Strategy: Generate random +/- 1 signals. These often have lower peak sidelobes
    # than continuous random signals for this specific metric.
    
    max_restarts = 200  # Adjust based on speed test, but keep conservative
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Generate random signs
        x = rng.choice([-1.0, 1.0], size=N)
        
        s = _score(x)
        
        if s < best_score:
            best_score = s
            best_x = x.copy()
            
            # If we found a significant improvement, do a quick local search
            # to refine it. Limit local search steps to prevent timeout.
            if time.time() < deadline - 1.0:
                # Simple local search: flip random bits, keep if better
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    idx = rng.integers(0, N)
                    x[idx] = -x[idx]
                    s2 = _score(x)
                    if s2 < best_score:
                        best_score = s2
                        best_x = x.copy()
                    else:
                        x[idx] = -x[idx] # revert
    
    return best_x
