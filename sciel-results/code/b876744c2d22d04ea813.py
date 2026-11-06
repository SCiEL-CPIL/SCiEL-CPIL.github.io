"""Narrow admissible tuple: greedy maximum independent set on conflict graph."""
import math
from collections import defaultdict

N = 5000

def _primes(limit):
    """Return list of primes up to limit (inclusive)."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    n = N
    active = bytearray(b'\x01') * (n + 1)
    current_size = n + 1
    
    # Precompute primes up to n+1
    all_primes = _primes(n + 1)
    
    # We will iterate until no violating prime is found
    while True:
        violating_prime = None
        min_residue = -1
        min_count = float('inf')
        residue_counts = None
        
        # Check primes in increasing order
        for p in all_primes:
            if p > current_size:
                break
            
            # Count residues for this prime
            counts = [0] * p
            # Optimization: if we can detect early that all residues are covered, 
            # we still need the counts to find the min.
            # So we must count all.
            for i in range(n + 1):
                if active[i]:
                    counts[i % p] += 1
            
            # Check if any residue is 0
            if 0 in counts:
                # This prime is satisfied (some residue is missing)
                continue
            else:
                # All residues are covered. This is a violation.
                # We found the smallest violating prime.
                violating_prime = p
                # Find min residue
                min_count = min(counts)
                min_residue = counts.index(min_count)
                break
        
        if violating_prime is None:
            # Admissible
            result = [i for i in range(n + 1) if active[i]]
            return result
        
        # Remove all elements with residue min_residue mod violating_prime
        p = violating_prime
        r = min_residue
        for i in range(n + 1):
            if active[i] and (i % p == r):
                active[i] = 0
                current_size -= 1
