"""Low discrepancy on all arithmetic progressions - local search with NumPy."""
import numpy as np
import time

N = 256
DEADLINE = 85  # seconds, safety margin before 90s kill

def _max_ap_discrepancy(signs):
    """Compute the maximum absolute signed sum over all arithmetic progressions in [0, N-1].
    signs: array of length N with values -1 or +1.
    Returns the maximum absolute value.
    """
    n = len(signs)
    max_abs = 0
    # For each step d from 1 to n-1
    for d in range(1, n):
        # For each start position s from 0 to n-1
        # The AP is: s, s+d, s+2d, ... while < n
        # We need to check all prefixes of each such AP
        # For efficiency, iterate over starts and accumulate
        for s in range(n):
            total = 0
            idx = s
            while idx < n:
                total += signs[idx]
                abs_total = abs(total)
                if abs_total > max_abs:
                    max_abs = abs_total
                # Early termination: if we've already exceeded a known bound, we could stop
                # But we need the true max, so continue
                idx += d
    return max_abs

def _max_ap_discrepancy_fast(signs):
    """Faster computation using the fact that for a given step d,
    we can process all starts more efficiently."""
    n = len(signs)
    max_abs = 0
    for d in range(1, n):
        # For step d, the APs starting at positions 0, 1, ..., d-1 are the "chains"
        # that cover all elements. But actually, every position can be a start.
        # Let's just iterate all starts but use numpy for the inner sum.
        for s in range(n):
            # Get indices: s, s+d, s+2d, ...
            indices = range(s, n, d)
            # Compute prefix sums
            running = 0
            for idx in indices:
                running += signs[idx]
                a = running if running > 0 else -running
                if a > max_abs:
                    max_abs = a
    return max_abs

def _max_ap_discrepancy_vectorized(signs):
    """Use NumPy to compute all AP sums more efficiently."""
    n = len(signs)
    max_abs = 0
    for d in range(1, n):
        # For each start s, compute prefix sums of the AP
        # Group by residue class mod d
        for r in range(d):
            # Positions: r, r+d, r+2d, ...
            # But starts can be any position, not just residue classes
            # Actually, an AP with step d starting at s has elements s, s+d, s+2d, ...
            # The prefix sums are: signs[s], signs[s]+signs[s+d], ...
            # Let's just iterate over all starts
            pass
    
    # Fallback to simple loop but with early exit
    for d in range(1, n):
        for s in range(n):
            total = 0
            idx = s
            while idx < n:
                total += signs[idx]
                if total > max_abs:
                    max_abs = total
                elif -total > max_abs:
                    max_abs = -total
                idx += d
    return max_abs

def construct():
    start_time = time.time()
    n = N
    
    # Initial guess: use a Thue-Morse-like sequence or simple pattern
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map to signs: +1 if even parity, -1 if odd
    def thue_morse(n):
        # Count bits
        c = 0
        m = n
        while m:
            c += m & 1
            m >>= 1
        return 1 if c % 2 == 0 else -1
    
    signs = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    current_max = _max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search: try flipping single positions and accept if it improves
    # Use a simple hill-climbing approach with random restarts if stuck
    
    import random
    rng = random.Random(42)
    
    # Precompute for each position, which APs it's part of (for faster evaluation)
    # Actually, let's just do straightforward local search
    
    max_iterations = 500000
    iter_count = 0
    
    while iter_count < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:  # 5 second safety margin
            break
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        old_val = signs[pos]
        signs[pos] = -old_val
        
        new_max = _max_ap_discrepancy(signs)
        
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Revert
            signs[pos] = old_val
            # Small probability of accepting a non-improving move to escape local optima
            if rng.random() < 0.01:
                signs[pos] = -old_val  # Keep the flip
                current_max = new_max
            else:
                signs[pos] = old_val
        
        iter_count += 1
    
    return list(best_signs)
