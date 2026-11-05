"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
_DEADLINE = 300.0  # seconds wall-clock limit

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points):
    """Check if points form an admissible tuple."""
    n = len(points)
    if n == 0:
        return True
    primes = _primes(n)
    for p in primes:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    # Start from the full set [0, N] and greedily remove the least-populated
    # residue class for each violating prime until admissible.
    points = list(range(N + 1))
    
    # Precompute primes up to N+1 (max possible tuple size)
    max_primes = _primes(N + 1)
    
    changed = True
    while changed:
        if time.time() - start_time > _DEADLINE - 10:
            break
        changed = False
        n = len(points)
        if n == 0:
            break
        # Only need to check primes <= n
        relevant_primes = [p for p in max_primes if p <= n]
        for p in relevant_primes:
            # Build residue class counts
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                # All residue classes are covered; remove the least-populated one
                smallest_class = min(classes.values(), key=len)
                smallest_set = set(smallest_class)
                points = [v for v in points if v not in smallest_set]
                changed = True
                break
    
    # Final safety check: if still not admissible, keep removing
    # (shouldn't happen with the greedy approach, but guard just in case)
    while not _is_admissible(points):
        if time.time() - start_time > _DEADLINE - 5:
            break
        n = len(points)
        if n == 0:
            break
        relevant_primes = [p for p in max_primes if p <= n]
        for p in relevant_primes:
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                smallest_class = min(classes.values(), key=len)
                smallest_set = set(smallest_class)
                points = [v for v in points if v not in smallest_set]
                break
    
    return points
# EVOLVE-BLOCK-END
