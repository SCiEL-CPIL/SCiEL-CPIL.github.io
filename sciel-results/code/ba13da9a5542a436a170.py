"""Structural change: FFT-based convolution to count powerful-number pair sums."""
import time
import math
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    powerful.add(1)
    max_b = int(round(limit ** (1/3))) + 2
    
    # Precompute square-free status
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    for p in range(2, int(math.isqrt(max_b)) + 1):
        if sq_free[p]:
            p2 = p * p
            for m in range(p2, max_b + 1, p2):
                sq_free[m] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    deadline = time.time() + 140
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers
    # We need to choose a limit. With FFT, we can handle larger limits.
    # Powerful numbers up to L: count ~ L^(1/3) * something
    # Let's try a range that gives good coverage.
    # For N up to ~200M, we need powerful up to 100M.
    # For N up to ~400M, we need powerful up to 200M.
    
    # Let's estimate: powerful count up to L is roughly 2*L^(1/3) + ...
    # Up to 100M: ~2*464 + ... ~ 1000-2000
    # Up to 200M: ~2*584 + ... ~ 1500-3000
    # Up to 500M: ~2*793 + ... ~ 2500-5000
    # Up to 1B: ~2*1000 + ... ~ 4000-8000
    
    # FFT size: if max powerful is M, FFT size needs to be at least 2*M.
    # For M=1B, FFT size 2B = 2*10^9 which is too large for memory (16GB for float64).
    # For M=500M, FFT size 1B = 10^9, 8GB for float64. Still large.
    # For M=200M, FFT size 400M = 4*10^8, 3.2GB for float64. Manageable.
    # For M=100M, FFT size 200M = 2*10^8, 1.6GB for float64. Fine.
    
    # Let's use limit = 150_000_000 (150M) to get N up to 300M.
    limit = 150_000_000
    
    powerful = _generate_powerful(limit)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: FFT convolution
    # Create indicator array
    max_val = powerful[-1]
    fft_size = 1 << (2 * max_val - 1).bit_length()  # next power of 2 >= 2*max_val
    # Actually, we need sums up to 2*limit, so fft_size >= 2*limit
    fft_size = 1 << (2 * limit).bit_length()
    
    # Use float32 to save memory
    f = np.zeros(fft_size, dtype=np.float32)
    for p in powerful:
        f[p] = 1.0
    
    # FFT
    F = np.fft.rfft(f)
    F2 = F * F
    conv = np.fft.irfft(F2, n=fft_size)
    
    # conv[s] should be approximately the number of pairs (a,b) with a+b=s
    # But we only want a <= b and both powerful.
    # conv[s] counts ALL ordered pairs (a,b) with a+b=s where both are powerful.
    # Number of unordered pairs with a <= b:
    # If s is even and s/2 is powerful: conv[s] = 2*(#pairs with a<b) + 1
    #   so #pairs with a<=b = (conv[s]+1)/2
    # If s is odd or s/2 not powerful: conv[s] = 2*(#pairs with a<b)
    #   so #pairs with a<=b = conv[s]/2
    
    # We want to find s that maximizes the number of pairs with a <= b.
    # This is equivalent to maximizing conv[s], with a tiebreaker for even s where s/2 is powerful.
    
    # Let's compute the "pair count" for each s
    # pair_count[s] = conv[s] // 2 if (s is odd or s//2 not in powerful)
    #                = (conv[s] + 1) // 2 if (s is even and s//2 in powerful)
    
    # For finding the max, we can just look at conv[s] values.
    # The maximum conv[s] will give us the best candidate.
    
    # Restrict to s >= 2 and s <= 2*limit
    valid_range = slice(2, 2 * limit + 1)
    conv_valid = conv[valid_range]
    
    if len(conv_valid) == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the index of the maximum
    best_idx = np.argmax(conv_valid)
    best_N = best_idx + 2
    best_conv_val = conv_valid[best_idx]
    
    # Time check
    if time.time() > deadline - 20:
        # Still need to generate pairs
        pass
    
    # Phase 3: Generate actual pairs for best_N
    powerful_set = set(powerful)
    pairs = []
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    if not pairs:
        # Fallback: find next best
        # Sort by conv value
        sorted_indices = np.argsort(conv_valid)[::-1]
        for idx in sorted_indices[:100]:
            N_candidate = idx + 2
            pairs = []
            half_N = N_candidate // 2
            for a in powerful:
                if a > half_N:
                    break
                b = N_candidate - a
                if b in powerful_set and b >= a:
                    pairs.append([a, b])
            if pairs:
                best_N = N_candidate
                break
    
    return {'N': best_N, 'pairs': pairs}
