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
        best_prime = None
        best_residue = -1
        best_count = float('inf')
        
        # Check all primes up to current_size to find the "cheapest" violation to fix
        for p in all_primes:
            if p > current_size:
                break
            
            # Count residues for this prime
            counts = [0] * p
            for i in range(n + 1):
                if active[i]:
                    counts[i % p] += 1
            
            # Check if any residue is 0
            if 0 in counts:
                # This prime is satisfied
                continue
            
            # All residues are covered. This is a violation.
            min_c = min(counts)
            
            # We want to remove the residue class with the minimum count
            # among all violating primes, to minimize the number of elements removed.
            if min_c < best_count:
                best_count = min_c
                best_prime = p
                best_residue = counts.index(min_c)
            elif min_c == best_count:
                # Tie-breaking: prefer larger prime? 
                # Removing from a larger prime affects a smaller fraction of the set?
                # Actually, if counts are equal, the impact on size is the same.
                # However, removing from a smaller prime might be "more disruptive" to other primes?
                # Let's stick to first found or prefer larger p to spread out removals?
                # Usually, smaller primes are more "constrained". 
                # Let's keep the first one (smallest p) to be consistent with previous behavior on ties, 
                # or perhaps prefer larger p to preserve small-prime structure?
                # For now, keep the first one encountered (smallest p) to minimize changes.
                pass
        
        if best_prime is None:
            # Admissible
            result = [i for i in range(n + 1) if active[i]]
            return result
        
        # Remove all elements with residue best_residue mod best_prime
        p = best_prime
        r = best_residue
        for i in range(n + 1):
            if active[i] and (i % p == r):
                active[i] = 0
                current_size -= 1
