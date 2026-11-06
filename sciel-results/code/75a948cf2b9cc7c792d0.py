"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np
from scipy.signal import fftconvolve

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful using exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count == 1:
                return False
        p += 1
    if temp > 1:
        return False  # remaining prime factor with exponent 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try increasing limits
    limits_to_try = [10**8, 1.2*10**8, 1.5*10**8]
    
    for limit in limits_to_try:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        powerful = _generate_powerful(limit)
        powerful_set = set(powerful)
        M = len(powerful)
        
        if M < 2:
            continue
        
        # Build indicator vector of length limit+1
        indicator = np.zeros(limit + 1, dtype=np.int32)
        for v in powerful:
            indicator[v] = 1
        
        # FFT convolution: conv[i] = number of ordered pairs (a,b) with a+b=i
        # Use fftconvolve for speed
        conv = fftconvolve(indicator, indicator, mode='full')
        conv = conv.astype(np.int64)
        
        # conv[k] = sum over a+b=k of indicator[a]*indicator[b]
        # This counts ordered pairs. For a < b: counted twice. For a == b: counted once.
        # We want unordered pairs with a <= b.
        # count_unordered(k) = (conv[k] + count_of_a_eq_b(k)) // 2
        # where count_of_a_eq_b(k) = 1 if k/2 is powerful and k is even, else 0
        
        # Focus on range [2, limit]
        valid_start = 2
        valid_end = limit
        if valid_end >= len(conv):
            valid_end = len(conv) - 1
        
        conv_valid = conv[valid_start:valid_end + 1]
        
        # Compute unordered pair counts
        # For each k in [2, limit], unordered_count = (conv[k] + self_pair) // 2
        # where self_pair = 1 if k is even and k//2 is powerful
        ks = np.arange(valid_start, valid_end + 1)
        
        # self_pair: k even and k//2 in powerful_set
        # Vectorize: check even and membership
        half = ks // 2
        even_mask = (ks % 2 == 0)
        
        # For self-pair, we need k//2 to be powerful
        # Build a set lookup - but vectorized membership is tricky
        # Alternative: create a boolean array for powerful membership
        max_half = (valid_end) // 2
        self_pair_arr = np.zeros(len(ks), dtype=np.int64)
        # Use numpy searchsorted on sorted powerful
        powerful_arr = np.array(powerful, dtype=np.int64)
        for idx in np.where(even_mask)[0]:
            h = half[idx]
            if h <= max_half:
                # binary search
                pos = np.searchsorted(powerful_arr, h)
                if pos < len(powerful_arr) and powerful_arr[pos] == h:
                    self_pair_arr[idx] = 1
        
        unordered_counts = (conv_valid + self_pair_arr) // 2
        
        best_idx = int(np.argmax(unordered_counts))
        best_N = int(ks[best_idx])
        best_count_here = int(unordered_counts[best_idx])
        
        if best_count_here > best_count:
            best_count = best_count_here
            pairs = []
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b in powerful_set and a <= b:
                    pairs.append([a, b])
            best_result = {'N': best_N, 'pairs': pairs}
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
    
    return best_result

# EVOLVE-BLOCK-END
