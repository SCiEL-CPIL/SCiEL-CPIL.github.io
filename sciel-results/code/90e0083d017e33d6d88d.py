"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    # Use FFT for efficient convolution
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    conv = np.fft.irfft(f * np.conj(f), n=2*n)
    # The autoconvolution is centered; we take the max absolute value
    return np.max(np.abs(conv))

def _integral_sq(x):
    """Compute the square of the integral (sum for discrete)."""
    s = np.sum(x)
    return s * s

def _score(x):
    """Compute the score: max_abs_autoconv / integral_sq. Lower is better."""
    denom = _integral_sq(x)
    if denom == 0:
        return np.inf
    return _autoconv_max_abs(x) / denom

def _random_init(n, seed):
    """Generate a random signed step function."""
    rng = np.random.RandomState(seed)
    # Random heights of +/-1
    x = rng.choice([-1.0, 1.0], size=n)
    return x

def _greedy_flip(x, max_passes=5, rng=None):
    """Greedy search: try flipping each element if it improves the score."""
    best_score = _score(x)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(len(x)):
            old_val = x[i]
            x[i] = -old_val
            new_score = _score(x)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                x[i] = old_val
    return x, best_score

def _simulated_annealing(x, start_temp=1.0, end_temp=0.01, steps=5000, rng=None):
    """Simulated annealing to find better configurations."""
    if rng is None:
        rng = np.random.RandomState()
    current_score = _score(x)
    best_x = x.copy()
    best_score = current_score
    
    for step in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Pick a random element to flip
        i = rng.randint(len(x))
        x[i] = -x[i]
        new_score = _score(x)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[i] = -x[i]  # revert
    
    return best_x, best_score

def _discrete_polish(x, max_evals=10000):
    """Final polish: try local discrete improvements with limited evaluations."""
    best_x = x.copy()
    best_score = _score(x)
    rng = np.random.RandomState(42)
    
    for _ in range(max_evals):
        i = rng.randint(len(best_x))
        candidate = best_x.copy()
        candidate[i] = -candidate[i]
        s = _score(candidate)
        if s < best_score:
            best_x = candidate
            best_score = s
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with a random initialization
    rng = np.random.RandomState(12345)
    x = _random_init(N, seed=12345)
    best_x = x.copy()
    best_score = _score(x)
    
    # Phase 1: Greedy passes
    x, score = _greedy_flip(x, max_passes=10, rng=rng)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 2: Simulated annealing
    x, score = _simulated_annealing(x, start_temp=1.0, end_temp=0.01, steps=20000, rng=rng)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 3: More greedy passes
    x, score = _greedy_flip(x, max_passes=20, rng=rng)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 4: Discrete polish
    x, score = _discrete_polish(x, max_evals=5000)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    return best_x
# EVOLVE-BLOCK-END
