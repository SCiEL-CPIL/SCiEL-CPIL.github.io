"""Program to find N with many representations as sums of two powerful numbers."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import numpy as np

def _sieve_primes(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using m^2 * b parametrization."""
    powerful = set()
    powerful.add(1)
    
    sqrt_limit = isqrt(limit)
    # We need primes up to sqrt_limit for factoring m
    # But actually we can generate powerful numbers differently:
    # A powerful number is of the form prod(p_i^e_i) with e_i >= 2.
    # Equivalent: n = m^2 * b where b is squarefree and b | m.
    
    # Precompute primes up to sqrt_limit
    if sqrt_limit >= 2:
        primes = _sieve_primes(sqrt_limit)
    else:
        primes = []
    
    for m in range(1, sqrt_limit + 1):
        m_sq = m * m
        if m_sq > limit:
            break
        # Factor m to get distinct prime factors
        temp = m
        prime_factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                prime_factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            prime_factors.append(temp)
        
        # Enumerate all squarefree divisors b of m
        n_pf = len(prime_factors)
        for mask in range(1 << n_pf):
            b = 1
            for i in range(n_pf):
                if mask & (1 << i):
                    b *= prime_factors[i]
            val = m_sq * b
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Generate powerful numbers up to a limit
    # We want to maximize the number of representations
    # For N, we need a + b = N with a, b powerful and a <= b
    # So a ranges from 1 to N/2, and b = N - a must be powerful
    # The number of such pairs depends on how many powerful numbers are in [1, N/2]
    # whose complement N - a is also powerful
    
    # Strategy: generate powerful numbers up to L, then for each candidate N in a range,
    # count pairs. But O(P^2) is expensive.
    # 
    # Better: generate powerful numbers, then for each pair (i, j) with i <= j,
    # compute N = P[i] + P[j] and count. Use numpy for speed.
    
    # First, let's generate powerful numbers up to a reasonable limit
    # and see how many we get
    L = 80_000_000
    
    powerful = _generate_powerful(L)
    P = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to numpy array for fast operations
    arr = np.array(powerful, dtype=np.int64)
    
    # Count representations for each N
    # For pairs (i, j) with i <= j, N = arr[i] + arr[j]
    # We can do this in chunks to avoid memory issues
    # 
    # Total pairs: P*(P+1)/2. For P=10000, that's 50M pairs.
    # We'll process in chunks of rows.
    
    count_dict = defaultdict(int)
    chunk_size = 500
    
    for i_start in range(0, P, chunk_size):
        i_end = min(i_start + chunk_size, P)
        # For each i in [i_start, i_end), j ranges from i to P-1
        # N = arr[i] + arr[j] for j >= i
        # We can vectorize: for each i, N = arr[i] + arr[i:]
        
        if time.time() > deadline - 15:
            break
        
        for i in range(i_start, i_end):
            # j from i to P-1
            sums = arr[i] + arr[i:]
            # Count each sum
            # Use numpy unique with counts for speed
            unique_sums, counts = np.unique(sums, return_counts=True)
            for s, c in zip(unique_sums, counts):
                count_dict[int(s)] += int(c)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        if count_dict:
            best_N = max(count_dict, key=lambda n: count_dict[n])
            # Reconstruct pairs
            powerful_set = set(powerful)
            pairs = []
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b >= a and b in powerful_set:
                    pairs.append([a, b])
            return {'N': best_N, 'pairs': pairs}
        return {'N': 2, 'pairs': [[1, 1]]}
    
    if not count_dict:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(count_dict, key=lambda n: count_dict[n])
    
    # Reconstruct the pairs for best_N
    powerful_set = set(powerful)
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
