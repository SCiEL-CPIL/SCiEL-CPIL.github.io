"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Build a signed equal-width step function that minimizes
    # max|autoconvolution| / (integral)^2.
    #
    # Approach: use a low-discrepancy sequence (Halton) mapped to
    # a compact set of signed levels, then apply a small local search
    # to reduce the peak autoconvolution while maintaining a large integral.
    #
    # The key insight: by using a quasi-random distribution of signs and
    # magnitudes, we can spread the autoconvolution mass more evenly,
    # reducing the peak compared to a constant sequence.
    
    def halton(index, base):
        """Generate the index-th value in a Halton sequence."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    # Generate Halton sequence values for each position
    halton_vals = np.array([halton(i + 1, 2) for i in range(N)])
    
    # Map to signed levels: use a symmetric set of values
    # We'll use values in {-1, -0.5, 0.5, 1} based on quantiles
    # This creates a balanced distribution of signs and magnitudes
    
    # Sort and assign levels based on quartiles
    sorted_idx = np.argsort(halton_vals)
    levels = np.zeros(N)
    
    # Assign 4 levels: -1, -0.5, 0.5, 1
    quarter = N // 4
    for i in range(N):
        rank = sorted_idx[i]
        if i < quarter:
            levels[rank] = -1.0
        elif i < 2 * quarter:
            levels[rank] = -0.5
        elif i < 3 * quarter:
            levels[rank] = 0.5
        else:
            levels[rank] = 1.0
    
    # Local search: try flipping individual elements to reduce peak autoconvolution
    def compute_score(x):
        """Compute max|autoconv| / (sum x)^2."""
        if np.sum(x) == 0:
            return np.inf
        autoconv = np.convolve(x, x, mode='full')
        return np.max(np.abs(autoconv)) / (np.sum(x) ** 2)
    
    current = levels.copy()
    current_score = compute_score(current)
    
    # Greedy local search with limited iterations
    rng = np.random.default_rng(42)
    max_iters = 5000
    for _ in range(max_iters):
        # Pick a random position to flip sign
        idx = rng.integers(0, N)
        old_val = current[idx]
        current[idx] = -old_val
        new_score = compute_score(current)
        if new_score < current_score:
            current_score = new_score
        else:
            current[idx] = old_val
    
    return current
# EVOLVE-BLOCK-END
