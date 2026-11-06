"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import numpy as np

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    deadline = time.time() + 280  # 300s limit, 20s safety margin
    
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    
    # Get all primes up to N (max possible tuple size)
    all_primes = _primes(N)
    
    # Greedy removal: for each prime, if all residue classes are covered,
    # remove the least-populated class
    # Process primes in increasing order (smaller primes have tighter constraints)
    for p in all_primes:
        if time.time() > deadline:
            break
        if len(points) < p:
            break  # If tuple size < p, we can't cover all p classes
        
        # Count occurrences of each residue class mod p
        # Use numpy for speed
        arr = np.array(points)
        residues = arr % p
        # Count each residue class
        counts = np.bincount(residues, minlength=p)
        
        # Check if all p classes are covered
        if np.all(counts > 0):
            # Find the least-populated class
            min_count = np.min(counts)
            # Get all classes with minimum count
            min_classes = np.where(counts == min_count)[0]
            # Pick the first one (or we could be smarter)
            remove_class = min_classes[0]
            # Remove all points with this residue
            mask = residues != remove_class
            points = list(arr[mask])
    
    # Final verification and cleanup
    # Ensure the result is admissible
    final_primes = _primes(len(points))
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for p in final_primes:
            if len(points) < p:
                break
            arr = np.array(points)
            residues = arr % p
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                min_count = np.min(counts)
                min_classes = np.where(counts == min_count)[0]
                remove_class = min_classes[0]
                mask = residues != remove_class
                points = list(arr[mask])
                changed = True
                break
    
    return points
# EVOLVE-BLOCK-END
