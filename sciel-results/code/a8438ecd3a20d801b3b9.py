"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120."""
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # We need the smallest n with 120 | phi(n)
    # 120 = 2^3 * 3 * 5
    # phi(n) must be divisible by 8, 3, and 5.
    
    # Let's do a direct search from n=1 upward, computing phi(n) efficiently.
    # For each n, we can compute phi(n) using prime factorization.
    # But scanning from 1 up could be slow if the answer is large.
    
    # Alternative: use a sieve to compute phi for all numbers up to some limit.
    # Since we want the minimum n, we can sieve phi values up to a reasonable bound
    # and return the first n where phi(n) % 120 == 0.
    
    # Let's sieve up to 1000000 to be safe, but we can stop early.
    # Actually, let's think about what the minimum n could be.
    
    # phi(n) divisible by 120 means phi(n) >= 120 (unless phi(n)=0 which is impossible for n>=1)
    # So n must be large enough that phi(n) >= 120.
    # For n = p (prime), phi(p) = p-1. So p-1 divisible by 120 means p = 121, but 121 is not prime.
    # p = 241: phi(241) = 240, which is divisible by 120. So n=241 is a candidate.
    # But maybe there's a smaller n with composite structure.
    
    # Let's just sieve phi up to a reasonable limit and find the minimum.
    # Use a modified sieve of Eratosthenes to compute phi values.
    
    max_n = 1000000
    phi = list(range(max_n + 1))
    
    for i in range(2, max_n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, max_n + 1, i):
                phi[j] -= phi[j] // i
    
    # Now find the smallest n with phi[n] % 120 == 0
    for n in range(1, max_n + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback: should not reach here
    return max_n
# EVOLVE-BLOCK-END
