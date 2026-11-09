"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lfsr_sequence(length, poly, init_state):
    """Generate a binary sequence from an LFSR, mapped to ±1."""
    seq = np.zeros(length, dtype=np.int8)
    state = init_state
    # Generate enough bits to fill the sequence
    # Use a simple LFSR with the given polynomial
    for i in range(length):
        seq[i] = 1 if (state & 1) == 1 else -1
        # Compute feedback bit
        fb = 0
        for bit_pos in poly:
            fb ^= (state >> bit_pos) & 1
        state = (state >> 1) | (fb << (max(poly) + 1 - 1))
        # Keep state within reasonable bounds
        state &= (1 << (max(poly) + 1)) - 1
    return seq

def _compute_c3_bound(heights):
    """Compute the upper bound on C3 (third autocorrelation inequality).
    
    For a signed step function, C3 relates to the maximum periodic autocorrelation.
    The upper bound is max over all lags of the absolute autocorrelation value.
    """
    n = len(heights)
    # Compute periodic autocorrelation for all lags
    # R(k) = sum_i h[i] * h[(i+k) % n]
    max_abs_corr = 0
    for k in range(n):
        corr = np.dot(heights, np.roll(heights, k))
        abs_corr = abs(corr)
        if abs_corr > max_abs_corr:
            max_abs_corr = abs_corr
    return max_abs_corr

def construct():
    """Construct a signed step function using LFSR-based binary sequence.
    
    Returns a sequence of 400 signed step heights (±1) derived from an
    LFSR with a primitive polynomial, which has near-optimal autocorrelation
    properties.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Try several LFSR configurations and pick the one with best C3 bound
    # Use primitive polynomials of various degrees
    # For degree m, the m-sequence has length 2^m - 1
    # We take the first N=400 bits
    
    best_heights = None
    best_score = float('inf')
    
    # Primitive polynomials for LFSR (represented as list of bit positions)
    # x^12 + x^6 + x^4 + x + 1 is primitive
    # x^11 + x^9 + x^7 + x + 1
    # x^10 + x^7 + x^5 + x^2 + 1
    # x^13 + x^4 + x^3 + x + 1
    # x^14 + x^10 + x^6 + x + 1
    primitive_polys = [
        [12, 6, 4, 1, 0],   # degree 12
        [11, 9, 7, 1, 0],   # degree 11
        [10, 7, 5, 2, 0],   # degree 10
        [13, 4, 3, 1, 0],   # degree 13
        [14, 10, 6, 1, 0],  # degree 14
        [15, 11, 9, 7, 0],  # degree 15
        [16, 14, 13, 12, 0],# degree 16
    ]
    
    # Try different initial states for each polynomial
    for poly in primitive_polys:
        degree = max(poly)
        # Try a few initial states
        for init in [1, 3, 7, 15, 31, 63, 127, 255]:
            if time.time() > deadline - 5.0:
                break
            
            heights = _lfsr_sequence(N, poly, init)
            score = _compute_c3_bound(heights)
            
            if score < best_score:
                best_score = score
                best_heights = heights.copy()
        
        if time.time() > deadline - 5.0:
            break
    
    # If no good result found, fall back to a simple alternating pattern
    if best_heights is None:
        best_heights = np.ones(N, dtype=np.int8)
        best_heights[::2] = -1
    
    return best_heights
# EVOLVE-BLOCK-END
