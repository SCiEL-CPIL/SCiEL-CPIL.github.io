import time
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number by exact factorization."""
    if n <= 1:
        return True
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt == 1:
                return False
        d += 1
    if n > 1:
        return False  # remaining prime factor with exponent 1
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    max_y = int(limit ** (1/3)) + 2
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > limit:
            break
        max_x = isqrt(limit // y3)
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerful.add(n)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s kill
    
    # Phase 1: Generate powerful numbers up to a moderate limit
    # Use 10^7 to keep K manageable (~15k numbers)
    M = 10_000_000
    powerful_list = _generate_powerful(M)
    K = len(powerful_list)
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    powerful_set = set(powerful_list)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Fallback: use smaller set
        M2 = 1_000_000
        powerful_list = _generate_powerful(M2)
        K = len(powerful_list)
        powerful_arr = np.array(powerful_list, dtype=np.int64)
        powerful_set = set(powerful_list)
    
    # Phase 2: Use FFT-based convolution to count pair sums efficiently
    # This is O(M log M) instead of O(K^2)
    # Create a boolean array where index i is 1 if i is powerful
    max_N = 2 * M
    conv_arr = np.zeros(max_N + 1, dtype=np.float32)
    for p in powerful_list:
        conv_arr[p] = 1.0
    
    # FFT convolution to count pairs
    n_fft = 1
    while n_fft < 2 * (max_N + 1):
        n_fft *= 2
    
    fft_arr = np.fft.rfft(conv_arr, n=n_fft)
    fft_sq = fft_arr * fft_arr
    conv_result = np.fft.irfft(fft_sq, n=n_fft)
    
    # conv_result[i] gives the number of ordered pairs (a,b) with a+b=i
    # For unordered pairs with a<=b, we need to adjust
    # But for finding N with many representations, the count is proportional
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Fallback to simpler approach
        counts = {}
        for i in range(K):
            a = powerful_arr[i]
            if a > max_N // 2:
                break
            b_needed = max_N // 2 - a  # This won't work well
            # Just return a simple result
            break
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N with the most representations
    # conv_result[i] counts ordered pairs, so for a<=b pairs, 
    # the count is roughly conv_result[i] // 2 (plus adjustment for a=b)
    
    # We want to find N that maximizes the number of (a,b) pairs with a<=b and a+b=N
    # From the convolution, for each N, the number of ordered pairs is conv_result[N]
    # The number of unordered pairs with a<=b is:
    #   if N is even and N/2 is powerful: conv_result[N]//2 + 1
    #   else: conv_result[N]//2
    # But actually, conv_result[N] already counts all ordered pairs (a,b) with a+b=N
    # where a and b are both powerful. Since the set is symmetric, 
    # the number of unordered pairs is ceil(conv_result[N] / 2)
    
    # Let's just use conv_result[N] as a proxy and find the max
    # But we need to verify actual pairs
    
    # Find top candidates
    top_indices = np.argsort(conv_result[:max_N+1])[::-1][:100]
    
    best_N = None
    best_pairs = []
    best_count = 0
    
    for idx in top_indices:
        N = int(idx)
        if N < 2:
            continue
        
        # Find all pairs (a,b) with a<=b and a+b=N
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
    
    if best_N is None or best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Verify each pair with exact factorization
    verified_pairs = []
    for a, b in best_pairs:
        if _is_powerful(a) and _is_powerful(b):
            verified_pairs.append([a, b])
    
    if not verified_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': verified_pairs}
