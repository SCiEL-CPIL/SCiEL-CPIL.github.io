"""Optimized program for C3-prime task: minimize max|autoconv| / integral²."""
import numpy as np
import time

N = 400

def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution of x."""
    n = len(x)
    # Full autoconvolution via FFT
    size = 2 * n
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    # ac[0:n] is the autoconvolution (lags 0 to n-1)
    # Actually for real x, ac[0] = sum(x^2), ac[k] = sum(x[i]*x[i-k])
    # We want the full autoconvolution: lags -(n-1) to (n-1)
    # np.fft.irfft gives circular, but with zero-padding it's linear
    # ac[0] = lag 0, ac[1] = lag 1, ..., ac[n-1] = lag n-1
    # ac[n] = lag -(n-1), ..., ac[2n-1] = lag -1
    max_val = np.max(np.abs(ac))
    return max_val

def _integral_sq(x):
    """Squared integral (sum of elements squared for step function with unit width)."""
    return np.sum(x * x)

def _score(x):
    """Compute C3' = max|autoconv| / integral²."""
    max_ac = _autoconv_max_abs(x)
    int_sq = _integral_sq(x)
    if int_sq == 0:
        return float('inf')
    return max_ac / int_sq

def _greedy_flip_optimize(x, deadline):
    """Greedy sign-flip optimization to reduce max|autoconv|."""
    x = x.copy()
    current_score = _score(x)
    best_x = x.copy()
    best_score = current_score
    
    n = len(x)
    
    # Precompute autoconvolution for efficiency
    def _recompute_score_after_flip(x, i):
        """Quick score after flipping x[i]."""
        # This is expensive; for N=400, FFT autoconv is ~O(N log N)
        # Let's just recompute
        return _score(x)
    
    # Use FFT-based autoconv for speed
    size = 2 * n
    while True:
        if time.time() > deadline:
            break
        improved = False
        # Try each flip
        for i in range(n):
            if time.time() > deadline:
                break
            x[i] *= -1
            new_score = _score(x)
            if new_score < current_score - 1e-15:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
                improved = True
            else:
                x[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_x, best_score

def _random_start(seed):
    """Generate a random sign sequence."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _structured_start():
    """Generate a structured initial sequence (alternating)."""
    x = np.ones(N)
    x[1::2] = -1.0
    return x

# EVOLVE-BLOCK-START
def construct():
    """Return a sequence of N signed step heights minimizing C3'."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with a good initial guess
    # Alternating sequence has autoconv that decays, but let's try multiple starts
    starts = [
        _structured_start(),
        _random_start(42),
        _random_start(123),
        _random_start(999),
    ]
    
    best_x = None
    best_score = float('inf')
    
    for s in starts:
        if time.time() > deadline:
            break
        x, score = _greedy_flip_optimize(s, deadline)
        if score < best_score:
            best_score = score
            best_x = x
        if time.time() > deadline:
            break
    
    # If we still have time, do more random restarts
    seed_counter = 1000
    while time.time() < deadline:
        s = _random_start(seed_counter)
        seed_counter += 1
        x, score = _greedy_flip_optimize(s, deadline)
        if score < best_score:
            best_score = score
            best_x = x
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
