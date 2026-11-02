"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_score(signs):
    """
    Compute the autoconvolution upper bound C3 for a signed step function.
    The signal is a step function with N equal intervals on [-1/4, 1/4].
    We want to minimize the peak sidelobe of the autocorrelation.
    
    The autocorrelation R[k] = sum_j x[j] * x[j+k] for lag k.
    The main lobe is at k=0 (energy). The upper bound C3 relates to the
    maximum of |R[k]| for k != 0, normalized appropriately.
    
    For a step function with heights h[j] in {-1, +1}, the autocorrelation
    at lag k is sum_j h[j]*h[j+k].
    
    The score is the max over non-zero lags of |R[k]| / N (or similar normalization).
    Lower is better.
    """
    x = signs.astype(np.float64)
    # Compute autocorrelation via FFT
    n = len(x)
    # Zero-pad to avoid circular convolution effects
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.fft(x, size)
    R = np.fft.ifft(F * np.conj(F))[:n].real
    # R[0] is the energy (N for signs), R[k] for k>0 is autocorrelation at lag k
    # We need the peak sidelobe: max of |R[k]| for k = 1, ..., n-1
    # By symmetry, R[k] = R[-k], so we look at k=1..n-1
    sidelobes = np.abs(R[1:])
    peak = np.max(sidelobes)
    # Normalize: the upper bound C3 is peak / N (or peak / sum(|x|^2))
    # For a step function with |h[j]|=1, energy = N
    score = peak / N
    return score


def _greedy_coordinate_descent(signs, max_flips=5000, time_limit=110.0):
    """
    Greedy coordinate descent: try flipping each sign, keep if it improves the score.
    Use FFT-based evaluation for O(N log N) per flip.
    """
    signs = signs.copy()
    current_score = _compute_score(signs)
    start_time = time.time()
    
    # Precompute FFT size
    n = N
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # We'll use a smarter approach: instead of recomputing full FFT each time,
    # we can use the fact that flipping one element changes the autocorrelation
    # in a structured way. But for simplicity and correctness, let's do full recomputation
    # but with early termination.
    
    improved = True
    flips_made = 0
    while improved and flips_made < max_flips:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try each position
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Flip sign at position i
            signs[i] = -signs[i]
            new_score = _compute_score(signs)
            if new_score < current_score - 1e-15:
                current_score = new_score
                improved = True
                flips_made += 1
            else:
                # Revert
                signs[i] = -signs[i]
    
    return signs, current_score


def _random_restarts(num_restarts, time_limit=100.0):
    """
    Perform multiple random restarts of greedy coordinate descent.
    """
    best_signs = None
    best_score = np.inf
    start_time = time.time()
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        # Random initial signs
        signs = np.random.choice([-1.0, 1.0], size=N)
        # Run greedy descent
        signs, score = _greedy_coordinate_descent(
            signs, 
            max_flips=2000,
            time_limit=time_limit - (time.time() - start_time)
        )
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
    
    return best_signs, best_score


def construct():
    """
    Construct a signed step function that minimizes the autoconvolution upper bound C3.
    Returns a sequence of N signed step heights (each in {-1, +1}).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Use random restarts with greedy coordinate descent
    # Start with a few restarts, each with a time budget
    num_restarts = 20
    
    best_signs, best_score = _random_restarts(num_restarts, time_limit=100.0)
    
    # If we have time left, do a final polish with more flips
    if best_signs is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 5.0:
            best_signs, best_score = _greedy_coordinate_descent(
                best_signs,
                max_flips=10000,
                time_limit=remaining
            )
    
    # Ensure we return an array of ints or floats in {-1, 1}
    return best_signs.astype(np.int8)

# EVOLVE-BLOCK-END
