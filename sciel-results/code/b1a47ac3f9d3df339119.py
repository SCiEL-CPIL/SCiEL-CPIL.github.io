"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
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
    Uses NumPy for fast residue counting; processes larger primes first to
    minimize total elements removed."""
    start_time = time.time()
    
    # Start with all integers in [0, N]
    points = np.arange(N + 1, dtype=np.int32)
    
    # Precompute primes up to N (max possible len), sorted descending
    all_primes = _primes(N)
    all_primes_desc = sorted(all_primes, reverse=True)
    
    # Iteratively repair: for each prime p <= current length, if all p residue
    # classes are present, remove the least-populated one
    changed = True
    while changed:
        if time.time() - start_time > _DEADLINE - 5:
            break
        changed = False
        current_len = len(points)
        # Only need to check primes up to current_len
        for p in all_primes_desc:
            if p > current_len:
                continue
            # Count residue classes mod p using NumPy
            residues = points % p
            counts = np.bincount(residues, minlength=p)
            if counts.min() > 0:
                # All residue classes covered; remove the least-populated one
                min_class = int(np.argmin(counts))
                # Remove all points with that residue
                points = points[points % p != min_class]
                changed = True
                break
    
    return points.tolist()
# EVOLVE-BLOCK-END
