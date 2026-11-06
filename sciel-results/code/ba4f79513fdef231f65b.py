"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
N = 5000
_DEADLINE = 300.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    """Start from full set [0, N], greedily remove least-populated residue class
    for each prime p <= len(points) that is fully covered, until admissible.
    
    Uses array-based counting for speed and processes primes in ascending order
    (smaller primes first) to maximize remaining set size.
    """
    start_time = time.time()
    
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    
    # Precompute primes up to N (max possible len)
    all_primes = _primes(N)
    
    # Iteratively repair: for each prime p <= current length, if all p residue
    # classes are present, remove the least-populated one
    changed = True
    while changed:
        if time.time() - start_time > _DEADLINE - 5:
            break
        changed = False
        current_len = len(points)
        # Only need to check primes up to current_len
        for p in all_primes:
            if p > current_len:
                break
            # Count residue classes mod p using an array for speed
            classes = [0] * p
            for v in points:
                classes[v % p] += 1
            
            # Check if all residue classes are covered
            if all(c > 0 for c in classes):
                # All residue classes covered; find the least-populated one
                min_class = min(range(p), key=lambda r: classes[r])
                # Remove all points with that residue
                points = [v for v in points if v % p != min_class]
                changed = True
                break
    
    return points
# EVOLVE-BLOCK-END
