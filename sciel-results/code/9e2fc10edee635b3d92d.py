"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_peak(x):
    """Compute the peak absolute value of circular autocorrelation excluding lag 0."""
    # x is a numpy array of ±1 values
    # Circular autocorrelation
    n = len(x)
    # Use FFT for efficiency
    X = np.fft.fft(x.astype(np.float64))
    power = np.abs(X) ** 2
    # Inverse FFT gives circular autocorrelation
    corr = np.fft.ifft(power).real
    # Exclude lag 0 (which is always N)
    # The peak is the max of |corr[k]| for k != 0
    # But we should consider all lags except 0
    peak = np.max(np.abs(np.concatenate([corr[1:], corr[:-1]])))
    return peak

def _autoconv_peak_fast(x):
    """Faster computation using real FFT."""
    n = len(x)
    X = np.fft.rfft(x.astype(np.float64))
    power = np.abs(X) ** 2
    corr = np.fft.irfft(power, n=n)
    peak = np.max(np.abs(np.concatenate([corr[1:], corr[:-1]])))
    return peak

def _flip_bit(x, i):
    """Flip bit at position i, return new array and change in peak."""
    x[i] = -x[i]
    return x

def _local_search(x, max_iters=50000, time_limit=5.0):
    """Greedy local search: flip bits that reduce the peak."""
    start_time = time.time()
    best_peak = _autoconv_peak_fast(x)
    x = x.copy()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        current_peak = _autoconv_peak_fast(x)
        # Try flipping each bit
        for i in range(N):
            x[i] = -x[i]
            new_peak = _autoconv_peak_fast(x)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if new_peak < best_peak:
                    best_peak = new_peak
            else:
                x[i] = -x[i]
    return x, best_peak

def _simulated_annealing(x, time_limit=30.0, initial_temp=10.0, cooling=0.9999):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    x = x.copy()
    current_peak = _autoconv_peak_fast(x)
    best_peak = current_peak
    best_x = x.copy()
    temp = initial_temp
    n = len(x)
    
    while (time.time() - start_time) < time_limit and temp > 0.01:
        # Pick a random position to flip
        i = random.randint(0, n - 1)
        x[i] = -x[i]
        new_peak = _autoconv_peak_fast(x)
        delta = new_peak - current_peak
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_x = x.copy()
        else:
            x[i] = -x[i]
        
        temp *= cooling
    
    return best_x, best_peak

def _generate_m_sequence(N):
    """Generate an m-sequence-like pattern using a linear feedback shift register."""
    # For N=400, we can use a primitive polynomial approach
    # But 400 is not of the form 2^m - 1, so we'll use a different approach
    # Use a pseudo-random pattern with good autocorrelation properties
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N)
    return x

def _generate_barker_like(N):
    """Generate a Barker-like sequence (maximal autocorrelation properties)."""
    # Barker sequences of length 13 are known, but for 400 we need something else
    # Use a Zadoff-Chu-like sequence
    x = np.ones(N)
    for i in range(1, N):
        x[i] = 1 if (i * (i + 1)) % 2 == 0 else -1
    return x

def _generate_structured(N, seed=0):
    """Generate structured patterns with good autocorrelation."""
    patterns = []
    
    # Pattern 1: Alternating blocks
    for block_size in [2, 4, 8, 16, 32]:
        n_blocks = N // block_size
        pattern = np.zeros(N)
        for b in range(n_blocks):
            sign = 1 if b % 2 == 0 else -1
            pattern[b*block_size:(b+1)*block_size] = sign
        if len(pattern) == N:
            patterns.append(pattern)
    
    # Pattern 2: Based on quadratic residues
    x = np.ones(N)
    for i in range(N):
        # Use quadratic character
        x[i] = 1 if (i * i) % 3 != 0 else -1
    patterns.append(x)
    
    # Pattern 3: Pseudo-random with fixed seed
    rng = np.random.RandomState(seed)
    patterns.append(rng.choice([-1, 1], size=N))
    
    # Pattern 4: Based on modular arithmetic
    x = np.ones(N)
    for i in range(N):
        x[i] = 1 if (i % 7) < 4 else -1
    patterns.append(x)
    
    # Pattern 5: Walsh-Hadamard based
    x = np.ones(N)
    for i in range(N):
        # Count set bits in binary representation
        bits = bin(i).count('1')
        x[i] = 1 if bits % 2 == 0 else -1
    patterns.append(x)
    
    return patterns

def construct():
    """Construct a signed step function that minimizes the autoconvolution peak."""
    start_time = time.time()
    time_budget = 145.0  # Safety margin before 150s limit
    
    # Generate initial patterns
    patterns = _generate_structured(N, seed=42)
    patterns.append(_generate_m_sequence(N))
    patterns.append(_generate_barker_like(N))
    
    # Also add some random patterns
    rng = np.random.RandomState(123)
    for _ in range(5):
        patterns.append(rng.choice([-1, 1], size=N))
    
    # Evaluate all patterns and pick the best
    best_x = None
    best_peak = float('inf')
    
    for p in patterns:
        p = p.astype(np.float64)
        peak = _autoconv_peak_fast(p)
        if peak < best_peak:
            best_peak = peak
            best_x = p.copy()
    
    # Local search on the best pattern
    best_x, best_peak = _local_search(best_x, time_limit=10.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget - 30:
        return best_x
    
    # Simulated annealing for further improvement
    remaining_time = time_budget - elapsed - 5
    if remaining_time > 5:
        best_x, sa_peak = _simulated_annealing(best_x, time_limit=remaining_time, initial_temp=5.0)
        if sa_peak < best_peak:
            best_peak = sa_peak
    
    # Final local search polish
    elapsed = time.time() - start_time
    if elapsed < time_budget - 5:
        best_x, final_peak = _local_search(best_x, time_limit=min(10.0, time_budget - elapsed))
        if final_peak < best_peak:
            best_peak = final_peak
    
    return best_x
# EVOLVE-BLOCK-END
