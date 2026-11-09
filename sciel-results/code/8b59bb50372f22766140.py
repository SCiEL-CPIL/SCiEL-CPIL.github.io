"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
DEADLINE = 300.0  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    
    # Sieve primes up to N (we'll need primes up to len(points) which can be up to N+1)
    all_primes = _primes(N + 1)
    
    # Start with full set [0, N]
    points = list(range(N + 1))
    
    # Sort primes in descending order
    primes_desc = sorted(all_primes, reverse=True)
    
    # We need to ensure for every prime p <= len(points), at least one residue class mod p is missing
    # Strategy: iterate through primes, and for each prime that currently has all classes covered,
    # remove the least-populated residue class
    
    # We'll do multiple passes since removing points can affect which primes need fixing
    # Process larger primes first since they're cheaper to fix
    
    changed = True
    pass_count = 0
    while changed:
        changed = False
        pass_count += 1
        
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
        
        n = len(points)
        if n == 0:
            break
        
        # We need to ensure for every prime p <= n, at least one residue class mod p is missing
        # Process primes in descending order (cheapest fixes first)
        for p in primes_desc:
            if p > n:
                continue
            
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Count residue classes present
            classes = {}
            for v in points:
                r = v % p
                classes[r] = classes.get(r, 0) + 1
            
            if len(classes) < p:
                # Already admissible for this prime
                continue
            
            # All p classes are covered - need to remove one
            # Remove the class with the fewest points
            min_count = min(classes.values())
            # Find which residue class(es) have min count
            # Remove all points in the least-populated class
            # If tie, pick the one whose removal causes least damage? For now, just pick first min
            target_residues = [r for r, c in classes.items() if c == min_count]
            # Pick the first one (or we could be smarter)
            target = target_residues[0]
            
            # Remove all points with v % p == target
            before = len(points)
            points = [v for v in points if v % p != target]
            if len(points) < before:
                changed = True
    
    return points
# EVOLVE-BLOCK-END
