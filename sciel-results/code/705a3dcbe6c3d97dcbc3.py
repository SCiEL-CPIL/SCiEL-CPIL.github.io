"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(x):
    """Compute max|autoconv(x)| / (sum(x))^2. Lower is better."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf
    ac = np.fft.irfft(np.abs(np.fft.rfft(x))**2)
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / (S * S)


def _evaluate_signs(signs):
    """Evaluate the score for a given sign vector (+1/-1)."""
    x = signs.astype(np.float64)
    return _compute_score(x)


def construct():
    start_time = time.time()
    deadline = 115.0  # seconds, with safety margin
    
    best_score = np.inf
    best_signs = np.ones(N, dtype=np.int8)
    
    # Initialize with a structured pattern: alternating blocks
    # This creates some cancellation in autoconvolution
    block_size = 20
    signs = np.ones(N, dtype=np.int8)
    for i in range(0, N, block_size):
        if (i // block_size) % 2 == 1:
            signs[i:i+block_size] = -1
    
    best_score = _evaluate_signs(signs)
    best_signs = signs.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_signs.astype(np.float64)
    
    # Local search: sign flips
    # Use a priority queue-like approach: try flipping positions with largest impact
    rng = np.random.RandomState(42)
    
    # Precompute for speed
    def fast_score(signs):
        x = signs.astype(np.float64)
        S = np.sum(x)
        if abs(S) < 1e-12:
            return np.inf
        # Use FFT for autoconvolution
        ac = np.fft.irfft(np.abs(np.fft.rfft(x))**2)
        return np.max(np.abs(ac)) / (S * S)
    
    # Iterative local search with restarts
    max_iter = 500000
    current_signs = best_signs.copy()
    current_score = best_score
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        # Pick a random position to flip
        pos = rng.randint(N)
        old_val = current_signs[pos]
        current_signs[pos] = -old_val
        
        new_score = fast_score(current_signs)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
        else:
            # Revert
            current_signs[pos] = old_val
    
    # Try a few different structured initializations
    for block_size in [5, 10, 15, 25, 30, 50, 75, 100]:
        if time.time() - start_time > deadline:
            break
        
        signs = np.ones(N, dtype=np.int8)
        for i in range(0, N, block_size):
            if (i // block_size) % 2 == 1:
                signs[i:i+block_size] = -1
        
        s = fast_score(signs)
        if s < best_score:
            best_score = s
            best_signs = signs.copy()
    
    # Final polish: more aggressive local search around best
    if time.time() - start_time < deadline - 10:
        current_signs = best_signs.copy()
        current_score = best_score
        for iteration in range(200000):
            if time.time() - start_time > deadline:
                break
            pos = rng.randint(N)
            old_val = current_signs[pos]
            current_signs[pos] = -old_val
            new_score = fast_score(current_signs)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs.copy()
            else:
                current_signs[pos] = old_val
    
    return best_signs.astype(np.float64)
# EVOLVE-BLOCK-END
