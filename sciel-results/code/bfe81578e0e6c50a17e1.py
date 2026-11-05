"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import Counter

N = 5000
DEADLINE = 310  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    start_time = time.time()
    
    # Precompute primes up to N+1 (max possible length is N+1=5001)
    max_len = N + 1
    primes_list = _primes(max_len)
    
    # Start with all numbers
    points = set(range(N + 1))
    
    # We'll iteratively remove elements to satisfy admissibility
    # Strategy: process primes, for each violating prime choose the best residue class to remove
    
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE:
            break
        
        changed = False
        n = len(points)
        
        # Find all violating primes
        violating_primes = []
        for p in primes_list:
            if p > n:
                break
            
            # Check if all residue classes are covered
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            if min(counts) > 0:
                violating_primes.append((p, counts))
        
        if not violating_primes:
            break
        
        # For each violating prime, we need to remove one residue class
        # Choose the prime and class that minimizes total removals
        # Heuristic: pick the prime where removing the smallest class causes least damage
        
        best_p = None
        best_class = None
        best_removal_count = float('inf')
        
        for p, counts in violating_primes:
            # Try each residue class, estimate impact
            for r in range(p):
                removal_count = counts[r]
                # Prefer smaller removals, and primes where this is more "urgent"
                # Use a score: removal_count / p (average class size as proxy for damage)
                score = removal_count
                if score < best_removal_count:
                    best_removal_count = score
                    best_p = p
                    best_class = r
                    if score == 1:
                        break
            if best_removal_count == 1:
                break
        
        if best_p is not None:
            # Remove all elements in the chosen residue class
            remove_residue = best_class
            points = {v for v in points if v % best_p != remove_residue}
            changed = True
    
    return sorted(points)
# EVOLVE-BLOCK-END
