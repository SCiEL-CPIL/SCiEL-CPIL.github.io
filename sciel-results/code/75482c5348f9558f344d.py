import numpy as np
import time

N = 400
DEADLINE = 140  # seconds, safety margin from 150s limit

def _autoconv(x):
    """Compute linear autoconvolution."""
    n = len(x)
    c = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        if k < n:
            c[k] = np.dot(x[:n-k], x[k:])
        else:
            c[k] = np.dot(x[k-n+1:], x[:2*n-1-k])
    return c

def _score(x):
    """Compute max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return np.max(np.abs(c)) / (s * s)

def _try_binary_local_search(x, max_iters=5000):
    """Local search on binary {-1,1} sequences."""
    x = x.copy()
    current_score = _score(x)
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        i = np.random.randint(N)
        old = x[i]
        x[i] = -x[i]
        new_score = _score(x)
        if new_score < current_score:
            current_score = new_score
        else:
            x[i] = old
    return x, current_score

def construct():
    start_time = time.time()
    # Start with all-ones (known optimal for real vectors)
    best_x = np.ones(N)
    best_score = _score(best_x)  # Should be 1/N = 0.0025
    
    # Try signed binary sequences with local search
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        # Initialize with random signs, biased toward +1 to keep sum large
        x = np.where(rng.random(N) < 0.7, 1.0, -1.0)
        x, s = _try_binary_local_search(x, max_iters=2000)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # If no improvement found, return the proven optimal all-ones
    return best_x
