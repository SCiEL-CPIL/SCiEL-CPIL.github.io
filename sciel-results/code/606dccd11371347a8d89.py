"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(x):
    """Linear autocorrelation via FFT. Returns array of length 2N-1."""
    n = len(x)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT efficiency
    fsize = 1
    while fsize < size:
        fsize <<= 1
    fx = np.fft.rfft(x, fsize)
    ac = np.fft.irfft(fx * np.conj(fx), fsize)[:size]
    return ac

def _score(x):
    """Compute max|AC| / (sum)^2. Return inf if sum is ~0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autocorr(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _initial_x():
    """Generate initial signed step function."""
    # Start with a pattern: mostly +1, with some -1 to reduce sidelobes
    # Use a simple construction: +1 for even indices, -1 for odd, then refine
    x = np.ones(N)
    # Introduce some -1s in a structured way
    # Use a Barker-like or random perturbation
    rng = np.random.RandomState(42)
    # Start with a chirp-like sign pattern
    for i in range(N):
        if (i * i % N) < N // 3:
            x[i] = -1
    return x

def _local_search(x, time_limit):
    """Local search: flip individual elements to reduce score."""
    start_time = time.time()
    current_x = x.copy()
    current_score = _score(current_x)
    best_x = current_x.copy()
    best_score = current_score
    
    n = len(current_x)
    rng = np.random.RandomState(123)
    
    # Precompute autocorrelation for fast updates
    # When we flip x[i] from a to b, the change in autocorrelation:
    # AC(k) changes by (b-a) * (x[i+k] + x[i-k]) for valid indices
    # This is complex; let's just recompute for simplicity with deadline guard
    
    max_iters = 50000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        # Pick a random index to flip
        i = rng.randint(n)
        old_val = current_x[i]
        new_val = -old_val  # flip sign
        if new_val == old_val:
            continue
        current_x[i] = new_val
        new_score = _score(current_x)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            current_x[i] = old_val
            # Occasionally accept worse moves (simulated annealing lite)
            if rng.random() < 0.05:
                current_x[i] = new_val
                current_score = new_score
    
    return best_x, best_score

def construct():
    deadline = time.time() + 100  # 100s safety margin
    
    # Try multiple initializations
    best_x = None
    best_score = np.inf
    
    # Initial candidate 1: structured pattern
    x1 = _initial_x()
    s1 = _score(x1)
    if s1 < best_score:
        best_score = s1
        best_x = x1.copy()
    
    # Initial candidate 2: all +1 (seed)
    x2 = np.ones(N)
    s2 = _score(x2)
    if s2 < best_score:
        best_score = s2
        best_x = x2.copy()
    
    # Initial candidate 3: random ±1
    rng = np.random.RandomState(7)
    x3 = rng.choice([1, -1], size=N)
    s3 = _score(x3)
    if s3 < best_score:
        best_score = s3
        best_x = x3.copy()
    
    # Initial candidate 4: block pattern (alternating blocks of +1 and -1)
    x4 = np.ones(N)
    block_size = 10
    for b in range(0, N, block_size * 2):
        end = min(b + block_size, N)
        x4[b:end] = -1
    s4 = _score(x4)
    if s4 < best_score:
        best_score = s4
        best_x = x4.copy()
    
    # Run local search with deadline guard
    remaining = deadline - time.time()
    if remaining > 5:
        best_x, best_score = _local_search(best_x, remaining - 5)
    
    # If we have more time, try another restart
    remaining = deadline - time.time()
    if remaining > 10:
        rng = np.random.RandomState(99)
        x5 = rng.choice([1, -1], size=N)
        s5 = _score(x5)
        if s5 < best_score:
            best_score = s5
            best_x = x5.copy()
        remaining2 = deadline - time.time()
        if remaining2 > 5:
            x5, s5 = _local_search(x5, remaining2 - 5)
            if s5 < best_score:
                best_score = s5
                best_x = x5.copy()
    
    return best_x
# EVOLVE-BLOCK-END
