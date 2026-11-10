import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0  # 600 seconds
    
    # We want a large N to approximate the limit a -> 1.
    # N=100,000 is a good balance between performance and accuracy.
    # The constraint M(x) <= 1.0001 for x in [1, 10N] requires M(y) <= 1.0001 for y in [2, 10].
    # M(1)=1, M(2)=0, M(3)=-1, M(4)=-1, M(5)=-2, M(6)=-1, M(7)=-2, M(8)=-2, M(9)=-3, M(10)=-2.
    # Max is 1, which is <= 1.0001. So any N works as long as we don't go beyond the range where M(y) spikes > 1.0001 for small y.
    # Actually, M(x) for x > N is M(floor(x/N)). Since x <= 10N, floor(x/N) <= 10.
    # So we just need M(y) <= 1.0001 for y in [1, 10].
    # M(1) = 1.
    # M(2) = 0.
    # M(3) = -1.
    # M(4) = -1.
    # M(5) = -2.
    # M(6) = -1.
    # M(7) = -2.
    # M(8) = -2.
    # M(9) = -3.
    # M(10) = -2.
    # All are <= 1.0001.
    
    N = 100000
    
    # Sieve of Eratosthenes to compute Möbius function
    mu = np.ones(N + 1, dtype=np.int8)
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Standard sieve for primes
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            # Mark multiples as not prime
            is_prime[i*i::i] = False
            # Update mu for multiples of i
            mu[i*i::i] *= -1
            # For multiples of i^2, mu should be 0
            # This step is tricky with vectorization. Let's use a more standard loop for mu or a specialized sieve.
            # A simpler way:
            # mu[n] = 0 if n has a squared prime factor.
            # mu[n] = (-1)^k if n has k distinct prime factors.
            
    # Let's use a more robust sieve for mu
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Linear sieve for Möbius function
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(N + 1, dtype=bool)
    
    for i in range(2, N + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
        
        if time.time() > deadline - 30:
            # If we are running out of time, reduce N? 
            # But the sieve is fast. 100k is instant.
            # Let's just break if we really need to, but it's unlikely.
            pass

    domain = np.arange(1, N + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # The evaluator will adjust f(1) to make sum(f(k)/k) = 0.
    # We can set f(1) to 0 initially.
    values[0] = 0.0
    
    return domain, values
