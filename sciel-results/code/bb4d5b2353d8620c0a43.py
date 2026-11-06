"""Combined program: vectorized greedy set-reduction with NumPy for speed."""
import time
import numpy as np

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


def construct():
    start_time = time.time()
    
    # Start with all integers in [0, N]
    points = np.arange(N + 1, dtype=np.int32)
    
    # Precompute primes up to N+1 (max possible len is N+1)
    all_primes = _primes(N + 1)
    
    # Main greedy repair loop
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE - 10:
            break
        changed = False
        current_len = len(points)
        
        # Process primes in ascending order (smaller primes first)
        # This removes fewer elements per step, preserving set size
        for p in all_primes:
            if p > current_len:
                break
            
            # Vectorized residue counting using NumPy
            residues = points % p
            # Count occurrences of each residue class
            class_counts = np.bincount(residues, minlength=p)
            
            # Check if all residue classes are covered
            if np.all(class_counts > 0):
                # Find the least-populated residue class
                min_class = np.argmin(class_counts)
                # Remove all points with that residue
                points = points[points % p != min_class]
                changed = True
                break
    
    # Final polish pass: re-check all primes up to len(points) one more time
    # to catch any remaining violations (should be none if loop converged,
    # but this ensures correctness)
    current_len = len(points)
    for p in all_primes:
        if p > current_len:
            break
        residues = points % p
        class_counts = np.bincount(residues, minlength=p)
        if np.all(class_counts > 0):
            min_class = np.argmin(class_counts)
            points = points[points % p != min_class]
            # After removal, length changed; restart check
            current_len = len(points)
    
    # Convert to list of distinct integers
    result = points.tolist()
    
    # Verify admissibility as a safety check
    n_final = len(result)
    for p in all_primes:
        if p > n_final:
            break
        residues = set(v % p for v in result)
        if len(residues) == p:
            # Should not happen, but if it does, do one more removal
            counts = [0] * p
            for v in result:
                counts[v % p] += 1
            min_c = min(counts)
            min_r = counts.index(min_c)
            result = [v for v in result if v % p != min_r]
            n_final = len(result)
    
    return result
