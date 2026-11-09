import time
import math

def construct():
    """
    Returns a list of distinct integers from 1 to 256 such that every pair is coprime,
    maximizing their sum.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill

    N = 256

    # Step 1: Find all primes up to N using Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Step 2: Sort primes in descending order to prioritize larger factors
    primes.sort(reverse=True)

    # Step 3: Greedy packing of primes into numbers (bins)
    # We want to maximize the sum of the products.
    # Merging primes p into a number x increases sum by x*(p-1).
    # To maximize gain, we merge into the largest possible x that doesn't exceed N.
    bins = []
    
    for p in primes:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_val = 0
        
        # Find the largest bin that can accept p
        for i, x in enumerate(bins):
            if x * p <= N:
                if x > best_val:
                    best_val = x
                    best_idx = i
        
        if best_idx != -1:
            bins[best_idx] = best_val * p
        else:
            bins.append(p)

    # Step 4: Add 1 to the list (1 is coprime with everything and distinct from others)
    result = bins + [1]
    
    # Ensure the result is a list of Python ints
    return [int(x) for x in result]
