"""Program using LFSR-generated binary sequence for signed step function to minimize C3 upper bound."""
import numpy as np

N = 400

# EVOLVE-BLOCK-START

# List of primitive polynomials (m, taps) where taps are indices of non-zero coefficients
# excluding the leading x^m term. Index 0 corresponds to x^0 (constant term).
# For a primitive polynomial, the constant term must be 1, so 0 is always in taps.
PRIMITIVE_POLYS = [
    # m=9
    (9, [0, 4]),
    (9, [0, 5]),
    (9, [0, 6, 7]),
    (9, [0, 3, 4]),
    (9, [0, 5, 6]),
    (9, [0, 1, 8]),
    (9, [0, 2, 8]),
    (9, [0, 3, 7]),
    (9, [0, 4, 7]),
    (9, [0, 6, 8]),
    (9, [0, 1, 4, 7]),
    (9, [0, 2, 5, 7]),
    (9, [0, 3, 5, 7]),
    (9, [0, 4, 6, 7]),
    # m=10
    (10, [0, 3]),
    (10, [0, 7]),
    (10, [0, 3, 8]),
    (10, [0, 4, 9]),
    (10, [0, 5, 9]),
    (10, [0, 6, 9]),
    (10, [0, 1, 9]),
    (10, [0, 2, 9]),
    (10, [0, 3, 7, 9]),
    (10, [0, 4, 7, 9]),
    (10, [0, 5, 7, 9]),
    (10, [0, 6, 7, 9]),
    (10, [0, 1, 3, 9]),
    (10, [0, 2, 3, 9]),
    (10, [0, 1, 6, 9]),
    (10, [0, 2, 6, 9]),
    # m=11
    (11, [0, 2]),
    (11, [0, 7]),
    (11, [0, 4, 9]),
    (11, [0, 5, 9]),
    (11, [0, 6, 9]),
    (11, [0, 7, 9]),
    (11, [0, 8, 9]),
    (11, [0, 1, 10]),
    (11, [0, 2, 10]),
    (11, [0, 3, 10]),
    (11, [0, 4, 10]),
    (11, [0, 5, 10]),
    (11, [0, 6, 10]),
    (11, [0, 8, 10]),
    (11, [0, 9, 10]),
    (11, [0, 1, 4, 9]),
    (11, [0, 2, 4, 9]),
    (11, [0, 3, 4, 9]),
    (11, [0, 1, 5, 9]),
    (11, [0, 2, 5, 9]),
    (11, [0, 3, 5, 9]),
    (11, [0, 4, 5, 9]),
    (11, [0, 6, 7, 9]),
    (11, [0, 1, 6, 9]),
    (11, [0, 2, 6, 9]),
    (11, [0, 3, 6, 9]),
    (11, [0, 4, 6, 9]),
    (11, [0, 5, 6, 9]),
    (11, [0, 7, 8, 9]),
    (11, [0, 1, 7, 9]),
    (11, [0, 2, 7, 9]),
    (11, [0, 3, 7, 9]),
    (11, [0, 4, 7, 9]),
    (11, [0, 5, 7, 9]),
    (11, [0, 6, 7, 9]),
    # m=12
    (12, [0, 6]),
    (12, [0, 11]),
    (12, [0, 7, 11]),
    (12, [0, 8, 11]),
    (12, [0, 9, 11]),
    (12, [0, 1, 11]),
    (12, [0, 2, 11]),
    (12, [0, 3, 11]),
    (12, [0, 4, 11]),
    (12, [0, 5, 11]),
    (12, [0, 6, 11]),
    (12, [0, 7, 11]),
    (12, [0, 8, 11]),
    (12, [0, 9, 11]),
    (12, [0, 10, 11]),
    (12, [0, 1, 4, 11]),
    (12, [0, 1, 6, 11]),
    (12, [0, 1, 7, 11]),
    (12, [0, 1, 8, 11]),
    (12, [0, 1, 9, 11]),
    (12, [0, 1, 10, 11]),
    (12, [0, 2, 4, 11]),
    (12, [0, 2, 6, 11]),
    (12, [0, 2, 7, 11]),
    (12, [0, 2, 8, 11]),
    (12, [0, 2, 9, 11]),
    (12, [0, 2, 10, 11]),
    (12, [0, 3, 4, 11]),
    (12, [0, 3, 6, 11]),
    (12, [0, 3, 7, 11]),
    (12, [0, 3, 8, 11]),
    (12, [0, 3, 9, 11]),
    (12, [0, 3, 10, 11]),
    (12, [0, 4, 6, 11]),
    (12, [0, 4, 7, 11]),
    (12, [0, 4, 8, 11]),
    (12, [0, 4, 9, 11]),
    (12, [0, 4, 10, 11]),
    (12, [0, 5, 6, 11]),
    (12, [0, 5, 7, 11]),
    (12, [0, 5, 8, 11]),
    (12, [0, 5, 9, 11]),
    (12, [0, 5, 10, 11]),
    (12, [0, 6, 7, 11]),
    (12, [0, 6, 8, 11]),
    (12, [0, 6, 9, 11]),
    (12, [0, 6, 10, 11]),
    (12, [0, 7, 8, 11]),
    (12, [0, 7, 9, 11]),
    (12, [0, 7, 10, 11]),
    (12, [0, 8, 9, 11]),
    (12, [0, 8, 10, 11]),
    (12, [0, 9, 10, 11]),
    # m=13
    (13, [0, 4]),
    (13, [0, 9]),
    (13, [0, 12]),
    (13, [0, 5, 12]),
    (13, [0, 6, 12]),
    (13, [0, 7, 12]),
    (13, [0, 8, 12]),
    (13, [0, 1, 12]),
    (13, [0, 2, 12]),
    (13, [0, 3, 12]),
    (13, [0, 4, 12]),
    (13, [0, 5, 12]),
    (13, [0, 6, 12]),
    (13, [0, 7, 12]),
    (13, [0, 8, 12]),
    (13, [0, 9, 12]),
    (13, [0, 10, 12]),
    (13, [0, 11, 12]),
    (13, [0, 1, 4, 12]),
    (13, [0, 1, 5, 12]),
    (13, [0, 1, 6, 12]),
    (13, [0, 1, 7, 12]),
    (13, [0, 1, 8, 12]),
    (13, [0, 1, 9, 12]),
    (13, [0, 1, 10, 12]),
    (13, [0, 1, 11, 12]),
    (13, [0, 2, 4, 12]),
    (13, [0, 2, 5, 12]),
    (13, [0, 2, 6, 12]),
    (13, [0, 2, 7, 12]),
    (13, [0, 2, 8, 12]),
    (13, [0, 2, 9, 12]),
    (13, [0, 2, 10, 12]),
    (13, [0, 2, 11, 12]),
    (13, [0, 3, 4, 12]),
    (13, [0, 3, 5, 12]),
    (13, [0, 3, 6, 12]),
    (13, [0, 3, 7, 12]),
    (13, [0, 3, 8, 12]),
    (13, [0, 3, 9, 12]),
    (13, [0, 3, 10, 12]),
    (13, [0, 3, 11, 12]),
    (13, [0, 4, 5, 12]),
    (13, [0, 4, 6, 12]),
    (13, [0, 4, 7, 12]),
    (13, [0, 4, 8, 12]),
    (13, [0, 4, 9, 12]),
    (13, [0, 4, 10, 12]),
    (13, [0, 4, 11, 12]),
    (13, [0, 5, 6, 12]),
    (13, [0, 5, 7, 12]),
    (13, [0, 5, 8, 12]),
    (13, [0, 5, 9, 12]),
    (13, [0, 5, 10, 12]),
    (13, [0, 5, 11, 12]),
    (13, [0, 6, 7, 12]),
    (13, [0, 6, 8, 12]),
    (13, [0, 6, 9, 12]),
    (13, [0, 6, 10, 12]),
    (13, [0, 6, 11, 12]),
    (13, [0, 7, 8, 12]),
    (13, [0, 7, 9, 12]),
    (13, [0, 7, 10, 12]),
    (13, [0, 7, 11, 12]),
    (13, [0, 8, 9, 12]),
    (13, [0, 8, 10, 12]),
    (13, [0, 8, 11, 12]),
    (13, [0, 9, 10, 12]),
    (13, [0, 9, 11, 12]),
    (13, [0, 10, 11, 12]),
]

def _generate_mls(taps, m):
    """Generate a maximal length sequence (MLS) of length 2^m - 1."""
    period = 2**m - 1
    # Initialize state with 1 followed by zeros
    state = np.zeros(m, dtype=np.int8)
    state[0] = 1
    
    seq = np.zeros(period, dtype=np.int8)
    
    for i in range(period):
        seq[i] = state[0]
        # Compute new bit
        # XOR of bits at tap positions
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        
        # Shift left
        state[1:] = state[:-1]
        state[0] = new_bit
        
    return seq

def construct():
    best_score = float('inf')
    best_heights = None
    
    # Pre-compute FFT size
    fft_size = 2 * N
    
    for m, taps in PRIMITIVE_POLYS:
        period = 2**m - 1
        if period < N:
            continue
            
        mls = _generate_mls(taps, m)
        # Map 0/1 to -1/1
        mls_pm1 = 2.0 * mls.astype(np.float64) - 1.0
        
        # Iterate over all possible starting phases
        # Optimization: Only check a subset of shifts if period is very large?
        # For m=13, period=8191. 8191 * ~150 polys = 1.2M correlations.
        # Using FFT might be faster or slower than np.correlate for N=400.
        # np.correlate is O(N^2) = 160k ops. FFT is O(N log N) ~ 400*9 = 3600 ops + overhead.
        # FFT overhead in Python/NumPy might be significant.
        # Let's stick to np.correlate but ensure we don't do it too many times.
        # Actually, let's use a stride trick or just simple slicing.
        
        # To speed up, we can skip some shifts? No, we want the best.
        # Let's just run it. 1.2M * 50us = 60s. Might be too slow.
        # Let's reduce the number of polynomials or shifts.
        # We'll keep all polynomials but maybe limit shifts to 1000 per poly?
        # Or, we can optimize the inner loop.
        
        for shift in range(period):
            # Extract window of length N
            idx = (np.arange(N) + shift) % period
            h = mls_pm1[idx]
            
            # Compute aperiodic autocorrelation
            ac = np.correlate(h, h, mode='full')
            
            # Exclude the center (lag 0)
            ac_non_peak = np.concatenate([ac[:N-1], ac[N:]])
            
            max_ac = np.max(np.abs(ac_non_peak))
            
            # Score is the upper bound on C3, which is max_ac / N
            score = max_ac / N
            
            if score < best_score:
                best_score = score
                best_heights = h.copy()
                
    if best_heights is None:
        # Fallback to random if no valid MLS found (should not happen)
        best_heights = np.random.choice([-1.0, 1.0], size=N)
        
    return best_heights

# EVOLVE-BLOCK-END
