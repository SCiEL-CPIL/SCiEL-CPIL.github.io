"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

def _greedy_construct(primes, random_seed=None):
    """
    Runs the greedy admissible tuple construction.
    If random_seed is not None, tie-breaking among min-count residue classes is random.
    """
    mask = np.ones(N + 1, dtype=bool)
    current_size = N + 1
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    for p in primes:
        if p > current_size:
            break
        
        active_indices = np.nonzero(mask)[0]
        if len(active_indices) == 0:
            break
        
        residues = active_indices % p
        counts = np.bincount(residues, minlength=p)
        
        if np.all(counts > 0):
            min_count = np.min(counts)
            # Find all classes with min count
            candidates = np.where(counts == min_count)[0]
            
            if len(candidates) > 1 and rng is not None:
                min_class = rng.choice(candidates)
            else:
                min_class = candidates[0]
            
            # Remove the chosen class
            # active_indices[residues == min_class] gives the global indices
            mask[active_indices[residues == min_class]] = False
            current_size -= min_count
            
    return np.nonzero(mask)[0].tolist()

def construct():
    primes = _primes(N + 1)
    
    # Try a few deterministic strategies and random tie-breakings
    best_result = []
    
    # 1. Original deterministic (min index)
    res = _greedy_construct(primes, random_seed=None)
    if len(res) > len(best_result):
        best_result = res
        
    # 2. Try a few random seeds to explore different tie-breaks
    # The number of runs is a trade-off between time and improvement.
    # Each run is fast (<1ms). 1000 runs is ~1s.
    for i in range(100):
        res = _greedy_construct(primes, random_seed=i)
        if len(res) > len(best_result):
            best_result = res
            
    return best_result
# EVOLVE-BLOCK-END
