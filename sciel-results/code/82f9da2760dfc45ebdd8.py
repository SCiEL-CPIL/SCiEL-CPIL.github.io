"""Find minimal n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Strategy:
    1. Compute phi for all numbers up to a reasonable limit using a sieve-like approach.
    2. Find the smallest n where phi(n) % 120 == 0.
    
    We know from number theory that the answer is small (around 143), so we can
    compute phi efficiently for numbers up to, say, 10000, which is very fast.
    """
    deadline = time.time() + 85  # 85 seconds safety margin
    
    # We'll compute phi(n) for n from 1 to some limit using a sieve method.
    # The answer is known to be small, so we don't need to go too far.
    # Let's compute up to 100000 to be safe, but we'll stop early once we find the answer.
    
    limit = 100000
    
    # Initialize phi array
    phi = list(range(limit + 1))
    
    # Sieve of Eratosthenes style to compute phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n with phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if n > 1 and phi[n] % 120 == 0:
            return n
        # Also check n=1: phi(1)=1, not divisible by 120
    
    # Fallback: return a known valid value
    return 143
# EVOLVE-BLOCK-END
