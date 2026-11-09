# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius_vectorized(K):
    """Compute the Möbius function mu(n) for n = 1..K using a vectorized approach."""
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    
    # Sieve of Eratosthenes style for Mobius
    # 1. Mark all as 1
    # 2. For each prime p, multiply mu[multiple] by -1
    # 3. For each prime p, set mu[multiple of p^2] to 0
    
    # We need primes first.
    # Standard sieve for primes
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(np.sqrt(K)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    # Initialize mu
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    
    for p in primes:
        # Mark multiples of p with -1
        # Indices: p, 2p, 3p, ...
        idx = np.arange(p, K + 1, p)
        mu[idx] *= -1
        
        # Mark multiples of p^2 with 0
        p2 = p * p
        if p2 <= K:
            idx2 = np.arange(p2, K + 1, p2)
            mu[idx2] = 0
            
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    
    # Fallback trivial solution: f(1) = 0
    def trivial_solution():
        return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)

    try:
        K = 200000
        
        # Check time
        if time.time() - start_time > 5:
            return trivial_solution()

        # Compute Mobius function
        mu = _compute_mobius_vectorized(K)
        
        if time.time() - start_time > 30:
            return trivial_solution()

        # Domain: indices where mu(k) != 0
        domain = np.where(mu[1:] != 0)[0] + 1
        # Base values: mu(k)
        base_values = mu[domain].astype(np.float64)
        
        # Calculate L = sum_{k=1}^K mu(k) log(k) / k
        # We want to maximize a = -c * L
        # L is expected to be negative.
        # We choose c to maximize -c*L subject to constraints.
        # Constraint: M(x) <= 1.0001.
        # For x <= K, M(x) = c. So c <= 1.0001.
        # To be safe against error terms for x > K, we use c = 0.99.
        
        c = 0.99
        
        # Apply scaling
        values = base_values * c
        
        # Final check
        if time.time() - start_time > 50:
            return trivial_solution()

        return domain, values

    except Exception:
        return trivial_solution()

# EVOLVE-BLOCK-END
