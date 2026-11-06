"""Rewritten program: start from full set and greedily remove least-populated residue classes."""
import time

N = 5000
DEADLINE = 300.0  # seconds, safety margin below 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes):
    """Check if the tuple is admissible: for every prime p <= len(points),
    not all residue classes mod p are covered."""
    n = len(points)
    for p in primes:
        if p > n:
            break
        covered = set()
        for v in points:
            covered.add(v % p)
            if len(covered) == p:
                return False
    return True

def construct():
    start_time = time.time()
    primes = _primes(N)
    
    # Start with the full set [0, 5000]
    points = list(range(N + 1))
    n = len(points)
    
    # We need to ensure admissibility for all primes p <= len(tuple)
    # Strategy: iteratively find a prime p <= len(points) where all p residue classes
    # are covered, and remove the least-populated residue class.
    
    # Precompute primes up to N+1 (since len(points) can be up to N+1)
    all_primes = _primes(N + 1)
    
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        changed = False
        n = len(points)
        
        # Find the first prime p <= n where all residue classes are covered
        for p in all_primes:
            if p > n:
                break
            
            # Count occurrences of each residue class mod p
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            
            # Check if all classes are covered
            if all(c > 0 for c in class_counts):
                # Find the least-populated class
                min_count = min(class_counts)
                min_class = class_counts.index(min_count)
                
                # Remove all points in that residue class
                points = [v for v in points if v % p != min_class]
                changed = True
                break
    
    return points
