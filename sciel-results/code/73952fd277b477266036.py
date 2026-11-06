"""Rewritten program for the suite task."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct pairwise coprime integers in 1..511 
    that maximizes the sum of 1/(512 - a).
    
    The optimal set consists of 1 and all primes less than 512.
    """
    # Sieve of Eratosthenes to find primes < 512
    limit = 512
    is_prime = [True] * limit
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit, i):
                is_prime[j] = False
    
    # Collect primes
    primes = [i for i in range(2, limit) if is_prime[i]]
    
    # Include 1
    result = [1] + primes
    
    return result
# EVOLVE-BLOCK-END
