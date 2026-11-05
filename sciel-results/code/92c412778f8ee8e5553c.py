import time
import sys
import numpy as np
from math import isqrt

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

    # Generate powerful numbers up to a limit that fits in memory for counting
    # L = 200,000,000 -> K ~ 30,000. K^2 ~ 9e8.
    # Counts array size 4e8 (int32) = 1.6 GB. Safe.
    pow_limit = 200_000_000

    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # Convert to numpy array
    pow_arr = np.array(powerful, dtype=np.int64)
    pow_set = set(powerful) # For pair extraction

    max_sum = 2 * pow_limit + 1
    
    # Use int32 for counts
    counts = np.zeros(max_sum, dtype=np.int32)

    # Chunked processing
    # K ~ 30,000. 
    # Chunk size 2000 -> 15 chunks.
    # Each chunk: 30,000 * 2,000 = 60,000,000 elements.
    # 15 * 60M = 900M elements processed total.
    chunk_size = 2000

    for i in range(0, K, chunk_size):
        chunk = pow_arr[i:i + chunk_size]
        # Compute outer sum
        sums = pow_arr[:, None] + chunk[None, :]
        
        # Bincount
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum)
        counts += chunk_counts

        # Time check
        if time.time() > deadline - 20:
            break

    if not any(counts > 0):
        return {'N': 5, 'pairs': [[1, 4]]}

    # Find the best N
    # We want to maximize the number of pairs.
    # counts[N] is the number of ordered pairs (a,b) such that a+b=N.
    # Since a,b are from the same set, and we want distinct pairs a<=b,
    # if N is even and N/2 is powerful, the unordered count is (counts[N] + 1) / 2.
    # If N is odd or N/2 is not powerful, it's counts[N] / 2.
    # Actually, counts[N] counts (a,b) and (b,a) separately if a!=b.
    # So the number of distinct pairs a<=b is roughly counts[N] / 2.
    # To maximize pairs, we maximize counts[N].
    
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])

    # Extract pairs
    pairs = []
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            pairs.append([a, b])

    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}

    return {'N': best_N, 'pairs': pairs}
