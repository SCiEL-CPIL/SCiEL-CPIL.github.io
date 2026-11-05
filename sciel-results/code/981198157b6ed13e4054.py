"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
from bisect import bisect_left, bisect_right
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form with sieve-based square-free check."""
    powerful = set()
    powerful.add(1)
    
    max_b = int(round(limit ** (1/3))) + 2
    
    # Precompute square-free status using sieve-like approach
    if max_b >= 1:
        sq_free = [True] * (max_b + 1)
        if max_b >= 0:
            sq_free[0] = False
        if max_b >= 1:
            sq_free[1] = True
        for p in range(2, int(math.isqrt(max_b)) + 1):
            if sq_free[p]:
                p2 = p * p
                for m in range(p2, max_b + 1, p2):
                    sq_free[m] = False
    
    for b in range(1, max_b + 1):
        if b > max_b:
            break
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
    deadline = time.time() + 145  # safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers up to a large limit
    limit = 80_000_000  # 8 * 10^7, enabled by FFT speedup
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    arr = np.array(powerful, dtype=np.int64)
    
    # Phase 2: Use FFT convolution to count all sums
    # Create indicator vector: f[i] = 1 if i is powerful, else 0
    # Then (f * f)[s] = number of pairs (a,b) with a+b=s where both are powerful
    # We need a <= b, so we'll adjust for double counting
    
    max_sum = 2 * limit
    # Use float32 for memory efficiency, with careful rounding
    # Pad to next power of 2 for FFT efficiency
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
    
    # Create indicator array
    f = np.zeros(n_fft, dtype=np.float32)
    for val in powerful:
        if val < n_fft:
            f[val] = 1.0
    
    # FFT convolution
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=n_fft)
    
    # Round to integers and take the relevant range
    # conv[s] = number of ordered pairs (a,b) with a+b=s, both powerful
    # For a <= b: count = (conv[s] + (1 if s is even and s/2 is powerful else 0)) / 2
    # Actually: conv[s] counts all ordered pairs. For a < b, they appear twice.
    # For a = b (i.e., s = 2a), they appear once.
    # So unordered pairs with a <= b: (conv[s] + delta) / 2 where delta = 1 if s/2 is powerful and s is even
    
    conv_int = np.round(conv[:max_sum + 1]).astype(np.int64)
    
    # Compute number of pairs with a <= b for each sum s
    # pairs[s] = (conv[s] + is_powerful(s//2) if s even) / 2
    # But we need to be careful: conv[s] = sum over a of f[a]*f[s-a]
    # This counts (a, s-a) and (s-a, a) separately when a != s-a
    # When a = s-a (i.e., s = 2a), it counts once
    
    # So: number of ordered pairs = conv[s]
    # Number of unordered pairs with a <= b = (conv[s] + I(s even and s/2 powerful)) / 2
    
    pairs_count = np.zeros(max_sum + 1, dtype=np.int64)
    
    # For even s where s/2 is powerful, add 1 before dividing
    even_s = np.arange(0, max_sum + 1, 2)
    half_s = even_s // 2
    # Check if half_s values are powerful
    # Use the f array: f[half_s] == 1 means half_s is powerful
    is_powerful_half = (f[half_s] == 1.0)
    
    adjusted = conv_int.copy()
    adjusted[even_s[is_powerful_half]] += 1
    
    pairs_count = adjusted // 2
    
    # Find the best N (maximum pairs count)
    # Exclude N=0 and N=1 (not valid since we need a+b=N with a,b >= 1)
    if max_sum >= 2:
        pairs_count[:2] = 0
    
    best_N = int(np.argmax(pairs_count))
    best_count = int(pairs_count[best_N])
    
    if best_count <= 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Time check: if we're running low, just return what we have
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Generate pairs quickly
        pairs = []
        for a in powerful:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in powerful_set and b >= a:
                pairs.append([a, b])
        if len(pairs) != best_count:
            best_count = len(pairs)
        return {'N': best_N, 'pairs': pairs}
    
    # Generate the actual pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    # Verify count matches
    if len(pairs) != best_count:
        # FFT rounding error; use actual count
        best_count = len(pairs)
    
    # If we have time, try to find an even better N by checking nearby candidates
    # or by increasing the limit and redoing
    remaining = deadline - time.time()
    if remaining > 40 and best_count < 20:
        # Try a larger limit
        limit2 = 120_000_000
        powerful2 = _generate_powerful(limit2)
        powerful_set2 = set(powerful2)
        K2 = len(powerful2)
        
        if K2 > K:
            arr2 = np.array(powerful2, dtype=np.int64)
            max_sum2 = 2 * limit2
            n_fft2 = 1
            while n_fft2 < max_sum2 + 1:
                n_fft2 *= 2
            
            f2 = np.zeros(n_fft2, dtype=np.float32)
            for val in powerful2:
                if val < n_fft2:
                    f2[val] = 1.0
            
            F2 = np.fft.rfft(f2)
            conv2 = np.fft.irfft(F2 * F2, n=n_fft2)
            conv2_int = np.round(conv2[:max_sum2 + 1]).astype(np.int64)
            
            even_s2 = np.arange(0, max_sum2 + 1, 2)
            half_s2 = even_s2 // 2
            is_powerful_half2 = (f2[half_s2] == 1.0)
            adjusted2 = conv2_int.copy()
            adjusted2[even_s2[is_powerful_half2]] += 1
            pairs_count2 = adjusted2 // 2
            if max_sum2 >= 2:
                pairs_count2[:2] = 0
            
            best_N2 = int(np.argmax(pairs_count2))
            best_count2 = int(pairs_count2[best_N2])
            
            if best_count2 > best_count:
                best_N = best_N2
                best_count = best_count2
                powerful = powerful2
                powerful_set = powerful_set2
                # Regenerate pairs
                pairs = []
                for a in powerful:
                    if a > best_N // 2:
                        break
                    b = best_N - a
                    if b in powerful_set and b >= a:
                        pairs.append([a, b])
                if len(pairs) != best_count:
                    best_count = len(pairs)
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
