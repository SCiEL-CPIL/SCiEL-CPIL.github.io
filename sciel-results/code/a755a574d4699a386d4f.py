"""Rewritten program: signed step heights minimizing max |autoconv| / (integral)^2."""
import numpy as np
import time
from scipy.fft import rfft, irfft

N = 400
SEED = 42

def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    # Linear autoconvolution via FFT
    n = len(x)
    # Use FFT size that is a power of 2 >= 2*n - 1 for efficiency
    fsize = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=fsize)
    conv = np.fft.irfft(X * np.conj(X), n=fsize)[:2 * n - 1]
    return np.max(np.abs(conv))

def _score(x):
    """Compute the score: max |autoconv| / (sum(x))^2. Lower is better."""
    integral = np.sum(x)
    if integral == 0:
        return np.inf
    return _autoconv_max_abs(x) / (integral ** 2)

def _generate_barker_like(n, seed=0):
    """Generate a binary sequence with good autocorrelation properties.
    
    Uses a construction inspired by Barker sequences and Gold codes.
    For arbitrary n, we use a pseudo-random sequence with local optimization
    to find a good starting point.
    """
    rng = np.random.RandomState(seed)
    x = rng.choice([-1, 1], size=n)
    
    # Quick local optimization to get a reasonable starting point
    best_score = _score(x)
    for _ in range(50):
        # Try flipping the element that most reduces max |autoconv|
        # Approximate: flip element with largest contribution to peak
        idx = rng.randint(n)
        old_val = x[idx]
        x[idx] = -old_val
        s = _score(x)
        if s < best_score:
            best_score = s
        else:
            x[idx] = old_val
    return x

def _local_optimize(x, max_iter=2000, deadline=None):
    """Local sign-flip optimization to minimize max |autoconv| / (sum)^2.
    
    Uses simulated annealing with adaptive temperature.
    """
    x = x.copy().astype(np.float64)
    n = len(x)
    best_x = x.copy()
    best_score = _score(x)
    current_score = best_score
    
    # Track time
    t0 = time.time()
    
    # Simulated annealing parameters
    T = 1.0
    cooling = 0.995
    
    for it in range(max_iter):
        if deadline is not None:
            if time.time() - t0 > deadline:
                break
        
        # Choose an index to flip
        # Weight toward indices that are part of high-convolution regions
        idx = np.random.randint(n)
        
        old_val = x[idx]
        x[idx] = -old_val
        new_score = _score(x)
        
        # Accept or reject
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / max(T, 1e-12)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[idx] = old_val
        
        T *= cooling
    
    return best_x, best_score

def _polish(x, deadline=None):
    """Final high-precision polish: try all single flips, keep best."""
    x = x.copy().astype(np.float64)
    n = len(x)
    best_score = _score(x)
    best_x = x.copy()
    
    t0 = time.time()
    
    # Multiple passes of greedy single-flip optimization
    for _ in range(3):
        if deadline is not None and time.time() - t0 > deadline:
            break
        improved = True
        while improved:
            if deadline is not None and time.time() - t0 > deadline:
                return best_x, best_score
            improved = False
            for i in range(n):
                if deadline is not None and time.time() - t0 > deadline:
                    return best_x, best_score
                old_val = x[i]
                x[i] = -old_val
                s = _score(x)
                if s < best_score:
                    best_score = s
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = old_val
    return best_x, best_score

def construct():
    """Construct a sequence of signed step heights minimizing max |autoconv| / (integral)^2."""
    t_start = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    np.random.seed(SEED)
    
    # Generate multiple candidate starting points
    candidates = []
    
    # Barker-like sequence
    x1 = _generate_barker_like(N, seed=42)
    candidates.append(x1)
    
    # Random sequence with some structure
    rng = np.random.RandomState(123)
    x2 = rng.choice([-1, 1], size=N)
    candidates.append(x2)
    
    # Alternating pattern with random perturbations
    x3 = np.ones(N) * (-1) ** np.arange(N)
    perturb_idx = rng.choice(N, size=N // 4, replace=False)
    x3[perturb_idx] *= -1
    candidates.append(x3)
    
    # Gold-code-like construction
    x4 = np.zeros(N, dtype=np.float64)
    for i in range(N):
        x4[i] = 1.0 if ((i * 7) % 13) < 6 else -1.0
    candidates.append(x4)
    
    # Evaluate candidates and pick the best
    best_x = None
    best_score = np.inf
    for x in candidates:
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    if time.time() - t_start > deadline:
        return best_x
    
    # Local optimization on the best candidate
    best_x, best_score = _local_optimize(best_x, max_iter=3000, deadline=deadline - (time.time() - t_start))
    
    if time.time() - t_start > deadline - 20:
        # Not enough time for polish, return current best
        return best_x
    
    # Polish pass
    best_x, best_score = _polish(best_x, deadline=deadline - (time.time() - t_start))
    
    return best_x

# For testing
if __name__ == "__main__":
    x = construct()
    print("Score:", _score(x))
    print("Shape:", x.shape)
    print("Values:", np.unique(x))
