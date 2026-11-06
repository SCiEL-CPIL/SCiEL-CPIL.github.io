import time
import sys
import numpy as np
from math import isqrt
from collections import defaultdict

def _sieve_primes(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0] = 0
    is_p[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]

def _is_square_free(n, primes):
    """Check if n is square-free using precomputed primes."""
    if n == 1:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
    return True

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number n satisfies: for every prime p|n, p^2|n.
    Equivalently, n = u^2 * v^3 where v is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # v must be square-free and v^3 <= limit
    max_v = int(round(limit ** (1.0 / 3.0))) + 2
    if max_v < 1:
        max_v = 1

    # Primes up to sqrt(max_v) for square-free check
    sieve_limit = isqrt(max_v) + 2
    if sieve_limit < 2:
        sieve_limit = 2
    primes = _sieve_primes(sieve_limit)
    
    for v in range(1, max_v + 1):
        if not _is_square_free(v, primes):
            continue
            
        v3 = v * v * v
        if v3 > limit:
            continue
            
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def _is_powerful(n):
    """
    Check if n is powerful using exact factorization.
    A number is powerful if every prime factor appears with exponent >= 2.
    """
    if n == 1:
        return True
    if n <= 0:
        return False
    
    t = n
    # Check factor 2
    if t % 2 == 0:
        exp = 0
        while t % 2 == 0:
            t //= 2
            exp += 1
        if exp == 1:
            return False
            
    p = 3
    while p * p <= t:
        if t % p == 0:
            exp = 0
            while t % p == 0:
                t //= p
                exp += 1
            if exp == 1:
                return False
        p += 2
        
    if t > 1:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    # Strategy: FFT convolution to count pair sums
    # Generate powerful numbers up to M, then use FFT to count how many
    # pairs (a,b) with a<=b sum to each value N.
    # 
    # Key insight: If we have K powerful numbers up to M, the FFT approach
    # counts ALL pairs in O(M log M) instead of O(K^2).
    # For M=100M, K~3000, K^2~9M which is feasible, but FFT on array of
    # size ~100M is faster and scales better.
    
    # Try different limits for the powerful number generation
    # The FFT array size needs to be at least 2*M (for sums up to 2M)
    # and should be a power of 2 or product of small primes for FFT efficiency
    
    limits_to_try = [
        100_000_000,    # 1e8 -> FFT size ~256M (2^28)
        50_000_000,     # 5e7
        20_000_000,     # 2e7
        10_000_000,     # 1e7
    ]
    
    for limit in limits_to_try:
        if time.time() > deadline - 30:
            break
        
        try:
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 25:
                continue
                
            K = len(powerful)
            if K < 2:
                continue
            
            # For FFT: we need an array where index = powerful number value
            # Then convolution of this array with itself gives counts of pairs
            # summing to each N.
            # But the array would be of size limit+1, which for 1e8 is 100MB.
            # That's feasible.
            
            # Actually, let's use a smarter approach:
            # Create a boolean array of size (2*limit + 1)
            # Mark powerful numbers
            # FFT-convolve with itself
            # The result at index N gives the number of ordered pairs (a,b) 
            # with a+b=N. For unordered pairs a<=b, we need to adjust.
            
            # Memory check: 2*limit+1 elements as float64 = 16*(2*limit+1) bytes
            # For limit=1e8: 16*2e8 = 3.2 GB - too much!
            # For limit=5e7: 1.6 GB - still too much
            # For limit=2e7: 640 MB - feasible
            # For limit=1e7: 320 MB - feasible
            
            # Let's be more careful about memory.
            # Use int32 for the indicator array, then convert to float64 for FFT.
            # Actually numpy's fft works on float64.
            
            # Alternative: use the fact that K is small (~sqrt(limit) or so)
            # and just do the O(K^2) loop but with numpy vectorization per row.
            # For K=5000, K^2/2 = 12.5M operations, very fast in numpy.
            # For K=10000, K^2/2 = 50M, still fast.
            # For K=20000, K^2/2 = 200M, might be slow.
            
            # Let's estimate K for each limit:
            # Number of powerful numbers up to x is ~ C * x^(1/3)
            # C is around 1-2
            # For x=1e8: K ~ 400-800
            # For x=2e7: K ~ 300-500
            # For x=1e7: K ~ 200-400
            
            # Wait, that seems low. Let me recalculate.
            # Powerful numbers: n = u^2 * v^3, v square-free
            # Count = sum_{v square-free, v^3 <= x} isqrt(x / v^3)
            # For v=1: isqrt(x) = sqrt(x)
            # For v=2: isqrt(x/8)
            # For v=3: isqrt(x/27)
            # ...
            # Dominated by v=1 term: ~sqrt(x)
            # So for x=1e8: K ~ 10000 + smaller terms ~ 12000
            # For x=2e7: K ~ 4500
            # For x=1e7: K ~ 3200
            
            # O(K^2) with numpy:
            # K=12000: K^2/2 = 72M pairs. In numpy, each row is a vector add + unique.
            # That's 12000 iterations, each doing a vector of average size 6000.
            # Total ~ 72M operations, should take a few seconds.
            
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Use numpy to count sums efficiently
            # For each i, compute sums = pow_arr[i] + pow_arr[i:]
            # This is O(K^2) total but vectorized per row
            
            counts = defaultdict(int)
            
            # Check if K is small enough for O(K^2) approach
            if K <= 20000:
                for i in range(K):
                    if time.time() > deadline - 15:
                        break
                    a = pow_arr[i]
                    sums = a + pow_arr[i:]
                    unique_sums, unique_counts = np.unique(sums, return_counts=True)
                    for s, c in zip(unique_sums, unique_counts):
                        counts[int(s)] += int(c)
            else:
                # Fallback: use FFT if K is too large
                # But we need to be careful about memory
                max_sum = int(pow_arr[-1]) * 2
                if max_sum <= 200_000_000:
                    # Use FFT
                    indicator = np.zeros(max_sum + 1, dtype=np.float64)
                    for p in pow_arr:
                        indicator[int(p)] = 1.0
                    
                    # FFT size should be power of 2 >= 2*(max_sum+1)
                    n_fft = 1
                    while n_fft < 2 * (max_sum + 1):
                        n_fft *= 2
                    
                    # This might be too large for memory
                    if n_fft > 536_870_912:  # 2^29
                        continue
                    
                    fft_indicator = np.fft.rfft(indicator, n=n_fft)
                    conv = np.fft.irfft(fft_indicator * fft_indicator, n=n_fft)
                    conv = conv[:max_sum + 1].astype(np.int64)
                    
                    # conv[n] counts ordered pairs (a,b) with a+b=n
                    # For unordered pairs a<=b:
                    # If n is even and n/2 is powerful, conv[n] includes (n/2, n/2) once
                    # Actually conv[n] = number of ordered pairs
                    # Unordered pairs a<=b: 
                    #   count = (conv[n] + (1 if n/2 is powerful and n even)) / 2
                    # But we just want to find the max, so let's use conv directly
                    # and adjust later
                    
                    # Find the N with maximum count
                    # Filter out N > 10^18 (not an issue here since max_sum <= 2e8)
                    # Also N >= 2
                    
                    valid_range = slice(2, min(max_sum + 1, 10**18 + 1))
                    best_N = 2 + np.argmax(conv[valid_range])
                    best_count_ordered = conv[best_N]
                    
                    # Convert to unordered pair count
                    # For a+b=N with a<=b:
                    # Each pair (a,b) with a!=b is counted twice in conv
                    # The pair (N/2, N/2) if N even and N/2 powerful is counted once
                    # So unordered_count = (conv[N] + delta) / 2 where delta = 1 if N/2 powerful and N even
                    if best_N % 2 == 0 and _is_powerful(best_N // 2):
                        best_count = (best_count_ordered + 1) // 2
                    else:
                        best_count = best_count_ordered // 2
                    
                    # Extract pairs
                    pow_set = set(powerful)
                    pairs = []
                    half_N = best_N // 2
                    end_idx = np.searchsorted(pow_arr, half_N, side='right')
                    
                    for a in pow_arr[:end_idx]:
                        b = best_N - a
                        if b in pow_set:
                            if _is_powerful(int(a)) and _is_powerful(int(b)):
                                pairs.append([int(a), int(b)])
                    
                    if pairs and len(pairs) > len(best_result['pairs']):
                        best_result = {'N': int(best_N), 'pairs': pairs}
                else:
                    continue
                    
        except Exception as e:
            continue
    
    # If we used the O(K^2) approach, extract the best
    if counts:
        best_N = max(counts, key=counts.get)
        best_count = counts[best_N]
        
        if best_N > 10**18:
            # Find next best
            sorted_items = sorted(counts.items(), key=lambda x: -x[1])
            for N, c in sorted_items:
                if N <= 10**18:
                    best_N = N
                    best_count = c
                    break
        
        if best_N >= 2:
            pow_set = set(powerful)
            pairs = []
            half_N = best_N // 2
            end_idx = np.searchsorted(pow_arr, half_N, side='right')
            
            for a in pow_arr[:end_idx]:
                b = best_N - a
                if b in pow_set:
                    if _is_powerful(int(a)) and _is_powerful(int(b)):
                        pairs.append([int(a), int(b)])
            
            if pairs and len(pairs) > len(best_result['pairs']):
                best_result = {'N': int(best_N), 'pairs': pairs}
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
