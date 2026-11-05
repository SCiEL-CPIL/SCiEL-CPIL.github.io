import time
import sys
import numpy as np
from math import isqrt
from collections import Counter

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using n = a^2 * b^3, b square-free."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0 / 3.0)) + 2

    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i * i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []

    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue

        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    # Strategy:
    # 1. Generate powerful numbers up to a moderate limit L (e.g., 50M).
    # 2. Use FFT to count sums of powerful numbers.
    # 3. Find the N with the maximum number of representations.
    # 4. Verify and extract pairs.
    
    # To maximize the number of pairs, we want a large set of powerful numbers.
    # However, FFT requires a dense array of size ~2*L.
    # L = 50,000,000 -> Array size 100,000,000.
    # FFT of size 100M is feasible but might be slow or memory heavy.
    # Let's try L = 20,000,000 first.
    # Number of powerful numbers up to 20M is roughly O(N^0.5) ~ 4000-5000.
    
    pow_limit = 20_000_000
    
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # Check time
    if time.time() > deadline - 30:
        # Fallback to a small case if generation is too slow (unlikely)
        return {'N': 5, 'pairs': [[1, 4]]}

    # Prepare array for FFT
    # We need to convolve the indicator function of powerful numbers with itself.
    # Let f[i] = 1 if i is powerful, 0 otherwise.
    # The convolution g[n] = sum f[i] * f[n-i] gives the number of ordered pairs (a,b) such that a+b=n.
    
    # Size of array must be at least 2 * pow_limit.
    # Next power of 2 for FFT efficiency.
    size = 1
    while size < 2 * pow_limit + 1:
        size *= 2
    
    # Create frequency array
    freq = np.zeros(size, dtype=np.float64)
    for p in powerful:
        freq[p] = 1.0
        
    # FFT
    freq_fft = np.fft.fft(freq)
    # Square the FFT (convolution in frequency domain)
    freq_fft_sq = freq_fft * freq_fft
    # Inverse FFT
    conv = np.fft.ifft(freq_fft_sq)
    conv = np.round(conv.real).astype(np.int64)
    
    # The number of unordered pairs (a,b) with a <= b and a+b = N is:
    # If N is even and N/2 is powerful, count = (conv[N] + 1) // 2
    # Else count = conv[N] // 2
    # We want to maximize this count.
    
    # We only care about N up to 2 * pow_limit
    max_N = 2 * pow_limit
    counts = conv[:max_N + 1]
    
    # Find the best N
    # We can just look for the max count, but we need to be careful about the a<=b condition.
    # Since conv[N] counts ordered pairs, and the set is symmetric, conv[N] is even if N/2 is not in the set,
    # and odd if N/2 is in the set.
    # The number of distinct pairs a<=b is ceil(conv[N] / 2).
    # So maximizing conv[N] is equivalent to maximizing the number of pairs.
    
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])
    
    # Check time before extraction
    if time.time() > deadline - 10:
        # If we're running out of time, return a known valid small pair?
        # No, we have a valid N and count. We just need to extract pairs.
        # Extraction might be slow if K is large.
        # Let's try to extract. If it fails, we return a fallback.
        pass

    # Extract pairs for best_N
    pow_set = set(powerful)
    pairs = []
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            pairs.append([a, b])
            
    if not pairs:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': best_N, 'pairs': pairs}
