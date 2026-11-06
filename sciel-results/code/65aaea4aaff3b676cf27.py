"""Improved program: greedy construction of largest admissible tuple in [0, 5000]."""
import time
import math
import numpy as np

N = 5000


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


def _is_admissible(points, max_prime_len):
    """Check if the tuple of points is admissible."""
    for p in _primes(max_prime_len):
        residues = set(x % p for x in points)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin out of 330s

    # Precompute primes up to N+1
    all_primes = _primes(N + 1)

    # Start with all points [0, N]
    points = np.arange(N + 1)

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        current_len = len(points)
        
        # Find all primes p <= current_len where all p residue classes are covered
        # and determine the least-populated class for each
        candidates = []  # list of (min_class_size, prime, min_residue)
        
        for p in all_primes:
            if p > current_len:
                break
            # Count residues using numpy
            residues = points % p
            # Count occurrences of each residue
            counts = np.bincount(residues, minlength=p)
            if len(counts) < p or np.min(counts) == 0:
                # Not all classes covered, or p > max residue (shouldn't happen)
                continue
            min_count = np.min(counts)
            if min_count == 0:
                continue
            # All p classes are covered; find the least populated
            min_residue = int(np.argmin(counts))
            candidates.append((min_count, p, min_residue))
        
        if not candidates:
            break
        
        # Sort by min_count ascending (remove cheapest first)
        candidates.sort(key=lambda x: x[0])
        
        # Remove the least-populated class for the best candidate
        min_count, p, min_residue = candidates[0]
        mask = (points % p) != min_residue
        points = points[mask]
        changed = True

    return sorted(points.tolist())
