import time
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def construct():
    # Greedy repair with NumPy-accelerated counting and a soft-violation
    # heuristic: instead of always processing the smallest violating prime first,
    # we prioritize the prime with the smallest min_count (most "balanced" coverage,
    # hence least damage when removing one residue class). This is a smooth surrogate
    # for the hard constraint satisfaction problem.
    
    primes = _primes(N)
    in_set = np.ones(N + 1, dtype=bool)
    current_len = N + 1
    
    # Precompute residue indices for each prime to speed up removal
    residue_indices = {}
    for p in primes:
        if p <= N:
            # For each residue r mod p, store the list of indices
            residue_indices[p] = [np.arange(r, N + 1, p) for r in range(p)]
    
    while time.time() < _DEADLINE:
        k = current_len
        if k < 2:
            break
        
        # Find all primes p <= k where all residues are covered
        # Use soft-violation score: min_count (lower = more balanced = less damage)
        best_p = None
        best_min_count = float('inf')
        best_target_residue = -1
        best_counts = None
        
        for p in primes:
            if p > k:
                break
            
            # Count residues using NumPy boolean indexing
            active = np.where(in_set)[0]
            if len(active) == 0:
                break
            residues = active % p
            counts = np.bincount(residues, minlength=p)
            
            min_count = int(np.min(counts))
            if min_count > 0:
                # All residues present - this prime is violated
                # Use min_count as soft-violation score (lower is "softer")
                if min_count < best_min_count:
                    best_min_count = min_count
                    best_p = p
                    best_target_residue = int(np.argmin(counts))
                    best_counts = counts
        
        if best_p is None:
            break
        
        # Remove all elements congruent to best_target_residue mod best_p
        p = best_p
        r = best_target_residue
        indices_to_remove = residue_indices[p][r]
        in_set[indices_to_remove] = False
        current_len -= int(np.sum(in_set[indices_to_remove] == False))  # count actual removals
        
        # Recount to be safe
        current_len = int(np.sum(in_set))
    
    # Build the final list
    result = [int(v) for v in np.where(in_set)[0]]
    return result
# EVOLVE-BLOCK-END
