"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(heights):
    """Compute the C3 upper bound for a signed step function.
    
    The C3 bound is based on the L3 norm of the aperiodic autocorrelation.
    For a step function with N intervals, the autocorrelation at lag k is:
    c(k) = sum_{i=0}^{N-1-k} h[i] * h[i+k]
    
    The C3 upper bound is (sum_{k=0}^{N-1} |c(k)|^3)^(1/3) / N
    (normalized by N to get a value comparable to the seed's 1.0)
    """
    n = len(heights)
    # Compute aperiodic autocorrelation efficiently
    # c(k) = sum_{i=0}^{n-1-k} h[i] * h[i+k] for k = 0, 1, ..., n-1
    # This is the convolution of h with reversed h, but only the first n terms
    # Actually, aperiodic autocorrelation: c(k) = sum_{i} h[i] * h[i+k]
    # We can compute this via FFT or direct computation
    # For N=400, direct is fine: O(N^2) = 160000 operations
    
    autocorr = np.zeros(n)
    for k in range(n):
        autocorr[k] = np.dot(heights[:n-k], heights[k:])
    
    # C3 bound: (sum |c(k)|^3)^(1/3) / N
    c3 = (np.sum(np.abs(autocorr)**3))**(1.0/3.0) / n
    return c3


def _try_pattern(pattern_type, seed_val=42):
    """Generate a pattern and compute its C3 bound."""
    if pattern_type == 'ones':
        h = np.ones(N)
    elif pattern_type == 'alternating':
        h = np.ones(N)
        h[1::2] = -1
    elif pattern_type == 'random':
        rng = np.random.RandomState(seed_val)
        h = rng.choice([-1, 1], size=N)
    elif pattern_type == 'random_block':
        # Random blocks of same sign
        rng = np.random.RandomState(seed_val)
        h = np.zeros(N)
        pos = 0
        while pos < N:
            block_len = rng.randint(1, 20)
            sign = rng.choice([-1, 1])
            end = min(pos + block_len, N)
            h[pos:end] = sign
            pos = end
    elif pattern_type == 'fourier':
        # Based on low-frequency Fourier modes
        t = np.arange(N)
        h = np.sign(np.sin(2 * np.pi * t / N))
        h[h == 0] = 1
    else:
        h = np.ones(N)
    
    return h, _compute_c3_bound(h)


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    best_h = np.ones(N)
    best_score = _compute_c3_bound(best_h)
    
    # Try simple patterns first
    simple_patterns = ['ones', 'alternating', 'fourier']
    for p in simple_patterns:
        if time.time() - start_time > deadline - 5:
            break
        h, score = _try_pattern(p)
        if score < best_score:
            best_score = score
            best_h = h
    
    # Try random sequences with various seeds
    # Each random eval takes ~0.1ms for N=400, so we can do many
    rng = np.random.RandomState(0)
    attempts = 0
    while time.time() - start_time < deadline - 10:
        # Generate random ±1 sequence
        h = rng.choice([-1, 1], size=N)
        score = _compute_c3_bound(h)
        if score < best_score:
            best_score = score
            best_h = h
        attempts += 1
        # Check time every 1000 attempts
        if attempts % 1000 == 0:
            if time.time() - start_time > deadline - 5:
                break
    
    # Try local search: flip random elements and see if it improves
    if time.time() - start_time < deadline - 5:
        current_h = best_h.copy()
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 50000
        
        while time.time() - start_time < deadline - 5 and no_improve_count < max_no_improve:
            # Pick a random position to flip
            idx = rng.randint(0, N)
            old_val = current_h[idx]
            current_h[idx] = -old_val
            
            new_score = _compute_c3_bound(current_h)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
                no_improve_count = 0
            else:
                current_h[idx] = old_val
                no_improve_count += 1
    
    return best_h
# EVOLVE-BLOCK-END
