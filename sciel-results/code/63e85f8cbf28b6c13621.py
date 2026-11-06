"""Program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 256
    # We want to assign signs s_i in {-1, 1} for i=1..n
    # to minimize max over all arithmetic progressions of |sum|
    
    # Strategy: Use a continuous relaxation with binary variables,
    # then round. Alternatively, use a greedy/local search approach.
    # Given the 60s constraint, let's use a smart initialization + local search.
    
    # First, let's try a deterministic construction based on Thue-Morse or similar,
    # then refine with local search.
    
    # Thue-Morse sequence: t(n) = parity of number of 1s in binary representation of n
    # Map to signs: (-1)^t(n)
    
    def thue_morse(n):
        # Count bits
        count = 0
        x = n
        while x:
            count += x & 1
            x >>= 1
        return count % 2
    
    # Initialize with Thue-Morse
    signs = np.array([(-1)**thue_morse(i) for i in range(1, n+1)], dtype=np.float64)
    
    # Now do local search: try flipping individual signs to reduce max discrepancy
    # We need an efficient way to compute max discrepancy over all APs
    
    # For each arithmetic progression (start, step, length), compute the sum
    # This is expensive if done naively. Let's precompute or use a smarter approach.
    
    # Alternative: For a given set of signs, compute the maximum absolute sum over all APs
    # We can iterate over all possible (start, step) pairs and for each, compute prefix sums
    # along that progression.
    
    def max_discrepancy(signs):
        """Compute maximum absolute signed sum over any arithmetic progression."""
        n = len(signs)
        max_abs = 0.0
        # Iterate over all possible steps
        for step in range(1, n + 1):
            # For each start position (0-indexed), compute sums along the AP
            for start in range(n):
                # The AP is start, start+step, start+2*step, ... < n
                # Compute cumulative sum
                curr = 0.0
                idx = start
                while idx < n:
                    curr += signs[idx]
                    if abs(curr) > max_abs:
                        max_abs = abs(curr)
                    idx += step
        return max_abs
    
    # Initial max discrepancy
    current_max = max_discrepancy(signs)
    
    # Local search: try flipping each sign and see if it reduces the max discrepancy
    # To make this efficient, we can use a simpler heuristic:
    # For each position, compute how much its sign contributes to the worst APs
    
    # Given time constraints, let's do a limited number of iterations
    # Try flipping each sign once per pass
    
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n):
            # Try flipping sign at position i
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = max_discrepancy(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                signs[i] = old_sign
    
    # Convert to integers
    result = [int(s) for s in signs]
    
    # Verify
    assert len(result) == n
    assert all(s in (-1, 1) for s in result)
    
    return result
# EVOLVE-BLOCK-END
