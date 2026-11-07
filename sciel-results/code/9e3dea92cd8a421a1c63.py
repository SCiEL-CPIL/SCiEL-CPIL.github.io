import math
import time
from collections import defaultdict
import numpy as np

def _sigma_from_factorization(n, factors):
    """
    Compute sigma(n) given the prime factorization of n.
    factors is a dict {prime: exponent}
    """
    if n == 1:
        return 1
    result = 1
    for p, e in factors.items():
        # Sum of geometric series: 1 + p + ... + p^e = (p^(e+1) - 1) / (p - 1)
        # Use integer arithmetic
        term = (p ** (e + 1) - 1) // (p - 1)
        result *= term
    return result

def _factorize(n, primes):
    """
    Factorize n using a list of primes.
    Returns a dict {prime: exponent}
    """
    factors = {}
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            factors[p] = exp
    if temp > 1:
        factors[temp] = 1
    return factors

def _generate_divisors(factors):
    """
    Generate all divisors from a factorization dict.
    """
    divisors = [1]
    for p, e in factors.items():
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
    return divisors

def _sieve_sigma_numpy(limit):
    """
    Compute sigma(n) for all n <= limit using NumPy.
    """
    sigma = np.zeros(limit + 1, dtype=np.int64)
    # The inner loop is vectorized
    for d in range(1, limit + 1):
        sigma[d::d] += d
    return sigma

def construct():
    start = time.time()
    deadline = start + 140.0

    # 1. Generate primes for factorization
    # We need primes up to sqrt(10^18) = 10^9? No, we only factorize M and its divisors.
    # M is around (10^7)^2 * log(10^7) ~ 10^14 * 16 ~ 1.6 * 10^15.
    # sqrt(M) ~ 4 * 10^7.
    # So we need primes up to ~10^7 for factorization of M and its divisors.
    # Let's generate primes up to 10^6 first, and extend if needed.
    # Actually, the sieve limit for sigma is the main cost.
    
    # Let's try L = 5,000,000 first to be safe on time, then expand if time permits.
    # The previous program used up to 10M. Let's aim for 10M if possible.
    
    L = 10_000_000
    
    # Check time budget for sieve
    # Sieve takes ~ L * sum(1/d) ~ L * log(L) operations.
    # NumPy vectorization makes this very fast.
    
    if time.time() > deadline - 10.0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # 2. Sieve sigma
    sigma = _sieve_sigma_numpy(L)
    
    # 3. Compute M_k = k * sigma(k)
    # k is 1..L
    ks = np.arange(1, L + 1, dtype=np.int64)
    # sigma array is 0-indexed, so sigma[1:] corresponds to k=1..L
    sigmas = sigma[1:]
    Ms = ks * sigmas
    
    # Filter out Ms > 10^18 (unlikely for L=10^7, M ~ 10^14, but just in case)
    # Ms are int64, 10^18 fits in int64 (max ~9.2*10^18)
    
    # 4. Find most frequent M
    # Sort Ms
    Ms_sorted = np.sort(Ms)
    
    # Find groups of equal values
    # We want the value with the highest frequency
    # Using np.diff to find boundaries
    # But for 10^7 elements, this is fast.
    
    # Alternative: use a dictionary on the sorted array?
    # Or just scan.
    
    # Let's find the mode.
    # Since the array is sorted, we can find runs of identical values.
    
    # Create an array of differences to find where values change
    # diff[i] = Ms_sorted[i] - Ms_sorted[i-1]
    # If diff[i] == 0, Ms_sorted[i] == Ms_sorted[i-1]
    
    # To avoid large memory allocation for diff, we can just scan.
    # But scanning 10^7 in Python is slow.
    # Let's use numpy to find indices where value changes.
    
    # indices where Ms_sorted[i] != Ms_sorted[i-1]
    # We prepend a dummy element to handle the first one
    # Actually, let's just use a trick:
    # unique_vals, counts, indices = np.unique(Ms, return_counts=True, return_index=True)
    # np.unique is O(N log N) and implemented in C, very fast.
    
    unique_Ms, counts, first_indices = np.unique(Ms, return_counts=True, return_index=True)
    
    # Find the index with max count
    max_idx = np.argmax(counts)
    best_M = int(unique_Ms[max_idx])
    best_count = int(counts[max_idx])
    
    # If the count is very low, we might want to try a larger L?
    # But we have limited time.
    # Let's proceed with this best_M.
    
    # 5. Refine: Find ALL k such that k*sigma(k) = best_M
    # We know some k from the sieve (those <= L).
    # But there might be k > L.
    # Also, we must ensure k <= sqrt(best_M).
    
    sqrt_M = math.isqrt(best_M)
    
    # Factorize best_M
    # We need primes up to sqrt(best_M).
    # best_M ~ 10^14, sqrt ~ 10^7.
    # We need a prime list up to 10^7.
    
    # Generate primes up to 10^7
    # Sieve of Eratosthenes
    prime_limit = 10_000_000
    is_prime = np.ones(prime_limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(prime_limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, prime_limit + 1) if is_prime[i]]
    
    # Factorize best_M
    factors_M = _factorize(best_M, primes)
    
    # Generate all divisors of best_M
    divisors = _generate_divisors(factors_M)
    
    # Filter divisors k such that k <= sqrt_M
    valid_ks = []
    for k in divisors:
        if k <= sqrt_M:
            # Check if k * sigma(k) == best_M
            # Compute sigma(k)
            # k is a divisor of best_M, so its prime factors are a subset of factors_M
            # We can factorize k quickly using factors_M
            factors_k = {}
            temp_k = k
            for p, e in factors_M.items():
                if temp_k % p == 0:
                    exp = 0
                    while temp_k % p == 0:
                        temp_k //= p
                        exp += 1
                    factors_k[p] = exp
            # temp_k should be 1 since k is a divisor of best_M
            if temp_k != 1:
                # This shouldn't happen if divisors are generated correctly
                # But just in case, factorize remaining
                factors_k_rem = _factorize(temp_k, primes)
                for p, e in factors_k_rem.items():
                    factors_k[p] = factors_k.get(p, 0) + e
            
            sigma_k = _sigma_from_factorization(k, factors_k)
            if k * sigma_k == best_M:
                valid_ks.append(k)
                
    # Sort valid_ks
    valid_ks.sort()
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return best found so far
        if not valid_ks:
             # Fallback
             return {'M': 333312, 'k': [336, 372, 434]}
        return {'M': best_M, 'k': valid_ks}
        
    # If we have time, can we do better?
    # The current best_M is the most frequent in the range 1..L.
    # Is it possible that a different M has more solutions, but none of them are in 1..L?
    # Unlikely, as solutions are clustered.
    # Is it possible that a different M has more solutions, and some are in 1..L, but we missed it because we only picked the global max?
    # We picked the global max count in 1..L.
    # If M1 has 5 solutions in 1..L and M2 has 4 solutions in 1..L, we pick M1.
    # What if M2 has 10 solutions total, but 6 are > L?
    # And M1 has 5 solutions total, all <= L?
    # Then M2 is better.
    # However, if k > L, then k is large.
    # M = k sigma(k) ~ k^2 log k.
    # If k > 10^7, M > 10^14 * 16 = 1.6 * 10^15.
    # The M's we found are around this magnitude.
    # It is possible that the "true" best M has solutions spread out.
    # But usually, collisions happen in the "dense" low range.
    
    # Let's stick with the best_M found.
    
    # Ensure types are correct
    best_M = int(best_M)
    valid_ks = [int(k) for k in valid_ks]
    
    if not valid_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': valid_ks}

# EVOLVE-BLOCK-END
