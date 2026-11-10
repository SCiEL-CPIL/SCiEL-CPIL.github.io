import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Use a boolean array for the set of active elements
    active = np.ones(N + 1, dtype=bool)
    primes = _primes(N)
    
    # Precompute indices for speed?
    # indices = np.arange(N + 1)
    
    for p in primes:
        # Check if we still need to satisfy this prime
        count = active.sum()
        if count < p:
            break
            
        # Get indices of active elements
        idx = np.nonzero(active)[0]
        
        # Calculate residues
        residues = idx % p
        
        # Count occurrences of each residue
        counts = np.bincount(residues, minlength=p)
        
        # Check if any residue is missed
        if counts.min() == 0:
            continue
            
        # Find the residue class with the minimum count
        min_r = np.argmin(counts)
        
        # Remove all active elements in this residue class
        # Create a mask for elements to remove
        # to_remove = np.arange(min_r, N + 1, p)
        # active[to_remove] = False
        
        # More efficient:
        # We know which indices in `idx` correspond to min_r
        # But `idx` is the list of active indices.
        # We can just update active array directly.
        remove_indices = np.arange(min_r, N + 1, p)
        active[remove_indices] = False

    # Convert back to list of integers
    return list(np.nonzero(active)[0])
