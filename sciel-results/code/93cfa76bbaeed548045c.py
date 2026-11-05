import numpy as np
import time
from itertools import product

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_autoconvolution_peak(signs):
    """Compute the L-infinity norm of the autoconvolution of the signed step function."""
    # The step function has N intervals each of width 1/N on [-1/4, 1/4]
    # Autoconvolution in continuous domain
    # For a step function, the autoconvolution can be computed via the discrete autocorrelation
    # scaled appropriately
    # The autoconvolution (f*f)(x) for a step function with heights h_i on intervals of width w
    # At lag k*w, the value is w^2 * sum_{i} h_i * h_{i+k} (for the overlapping part)
    # Actually for convolution (not correlation), we need to be careful about signs
    # Let's compute the discrete autoconvolution (correlation) which is standard
    n = len(signs)
    # Full autocorrelation
    ac = np.correlate(signs, signs, mode='full')
    # The peak is the max absolute value
    return np.max(np.abs(ac))

def _compute_c3_bound(signs):
    """Compute the C3 upper bound - trying different definitions."""
    # Definition 1: L-infinity norm of autoconvolution
    ac = np.correlate(signs, signs, mode='full')
    return np.max(np.abs(ac))

def _try_random_seeds(n_trials=5000, seed_base=42):
    """Try random ±1 sequences."""
    best_val = float('inf')
    best_signs = None
    rng = np.random.RandomState(seed_base)
    for i in range(n_trials):
        signs = rng.choice([-1, 1], size=N)
        val = _compute_c3_bound(signs)
        if val < best_val:
            best_val = val
            best_signs = signs.copy()
    return best_signs, best_val

def _try_zadoff_chu_like():
    """Try Zadoff-Chu-like sequences."""
    # Zadoff-Chu sequences have constant amplitude and zero autocorrelation for non-zero lags
    # But they're complex; for real ±1, we can try related constructions
    # For even N, we can try m-sequence based approaches
    best_val = float('inf')
    best_signs = None
    # Try simple structured patterns
    for period in [2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 31, 32, 37, 41, 47, 61, 64, 71, 73, 83, 89, 97, 101, 103, 107, 109, 113, 127, 128, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397]:
        if period > N:
            continue
        # Create a periodic pattern
        base = np.ones(period)
        # Try different sign patterns for the base
        # For small periods, enumerate
        if period <= 8:
            for mask in range(1 << period):
                base = np.array([1 if (mask >> i) & 1 else -1 for i in range(period)])
                signs = np.tile(base, N // period + 1)[:N]
                val = _compute_c3_bound(signs)
                if val < best_val:
                    best_val = val
                    best_signs = signs.copy()
        else:
            # For larger periods, try a few random patterns
            rng = np.random.RandomState(period)
            for _ in range(10):
                base = rng.choice([-1, 1], size=period)
                signs = np.tile(base, N // period + 1)[:N]
                val = _compute_c3_bound(signs)
                if val < best_val:
                    best_val = val
                    best_signs = signs.copy()
    return best_signs, best_val

def _try_local_search(initial_signs, n_iterations=5000):
    """Local search by flipping individual bits."""
    signs = initial_signs.copy().astype(float)
    current_val = _compute_c3_bound(signs)
    best_val = current_val
    best_signs = signs.copy()
    rng = np.random.RandomState(123)
    
    for i in range(n_iterations):
        # Flip a random position
        pos = rng.randint(N)
        signs[pos] *= -1
        new_val = _compute_c3_bound(signs)
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = signs.copy()
        else:
            signs[pos] *= -1  # revert
    return best_signs, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try different strategies
    candidates = []
    
    # Strategy 1: Random sequences
    if time.time() < deadline - 30:
        signs1, val1 = _try_random_seeds(n_trials=10000, seed_base=42)
        candidates.append((val1, signs1))
    
    # Strategy 2: Structured periodic sequences
    if time.time() < deadline - 20:
        signs2, val2 = _try_zadoff_chu_like()
        if signs2 is not None:
            candidates.append((val2, signs2))
    
    # Strategy 3: Local search from best candidate
    if candidates and time.time() < deadline - 10:
        best_val, best_signs = min(candidates, key=lambda x: x[0])
        signs3, val3 = _try_local_search(best_signs, n_iterations=20000)
        candidates.append((val3, signs3))
    
    if not candidates:
        # Fallback: return all ones
        return np.ones(N)
    
    # Return the best candidate
    best_val, best_signs = min(candidates, key=lambda x: x[0])
    return best_signs.astype(float)

# EVOLVE-BLOCK-END
