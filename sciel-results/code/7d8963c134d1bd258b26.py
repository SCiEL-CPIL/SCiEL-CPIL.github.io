"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import signal


def _autoconv_full(x):
    """Compute full autoconvolution of x."""
    return signal.convolve(x, x, mode='full')


def _score(x):
    """Compute the raw score: max |autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_full(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def _rudin_shapiro(n):
    """Generate Rudin-Shapiro sequence of length n (values ±1)."""
    # Rudin-Shapiro sequence
    a = [1]
    b = [1]
    while len(a) < n:
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        a = a_new
        b = b_new
    return a[:n].astype(np.float64)


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    candidates = []
    
    # Candidate 1: constant ones
    candidates.append(np.ones(N))
    
    # Candidate 2: constant -ones
    candidates.append(-np.ones(N))
    
    # Candidate 3: Rudin-Shapiro
    if time.time() < deadline:
        rs = _rudin_shapiro(N)
        candidates.append(rs)
    
    # Candidate 4: Alternating signs
    candidates.append(np.ones(N) * np.where(np.arange(N) % 2 == 0, 1, -1))
    
    # Candidate 5: Random signed sequences (a few)
    rng = np.random.RandomState(42)
    for i in range(5):
        if time.time() >= deadline:
            break
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates
    best_x = candidates[0]
    best_score = _score(best_x)
    
    for x in candidates[1:]:
        if time.time() >= deadline:
            break
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x
    
    return best_x
# EVOLVE-BLOCK-END
