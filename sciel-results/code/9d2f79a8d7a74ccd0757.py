import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Maximum key K
    K = 200000
    
    # Compute Mobius function mu(k) for k in 1..K
    mu = np.ones(K + 1, dtype=np.int8)
    # Sieve for Mobius
    # mu(n) = 0 if n is not square-free
    # mu(n) = (-1)^k if n is product of k distinct primes
    
    # Initialize mu to 1
    # Use a sieve
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                # This simple sieve is too slow for 200k? No, 200k is small.
                # But standard sieve is better.
    
    # Better Mobius sieve
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    # mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    # Domain and values
    # We include all k where mu(k) != 0
    # But the task allows any subset. Including all is fine.
    # However, if we include k where mu(k)=0, f(k)=0.
    # It's better to only include non-zero to save space?
    # m <= 200000 is fine.
    
    # Create domain array
    # We can just use 1..K
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Check time
    if time.time() > deadline - 10:
        return domain, values

    return domain, values
