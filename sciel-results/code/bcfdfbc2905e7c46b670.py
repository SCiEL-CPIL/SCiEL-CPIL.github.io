import numpy as np
import time
import random

N = 5000

def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return [int(p) for p in np.where(sieve)[0]]

def _greedy_repair(S, primes, randomize=False):
    """
    Repairs S to be admissible.
    Returns the modified S (boolean array).
    """
    changed = True
    while changed:
        changed = False
        current_size = np.sum(S)
        if current_size == 0:
            break
            
        for p in primes:
            if p > current_size:
                break
                
            indices = np.where(S)[0]
            if len(indices) == 0:
                break
                
            residues = indices % p
            counts = np.bincount(residues, minlength=p)
            
            if np.all(counts > 0):
                min_count = np.min(counts)
                # Find all classes with min count
                min_classes = np.where(counts == min_count)[0]
                
                if randomize and len(min_classes) > 1:
                    r_to_remove = min_classes[random.randint(0, len(min_classes) - 1)]
                else:
                    r_to_remove = min_classes[0]
                
                mask = (residues == r_to_remove)
                S[indices[mask]] = False
                changed = True
                # We don't break the inner loop, we continue to check next prime
                # because the removal might have affected it, or it might still be covered.
                # Actually, if we remove elements, a previously covered prime might become uncovered?
                # No, removing elements only decreases counts.
                # If a prime was covered (all counts > 0), it remains covered or becomes uncovered?
                # If we remove a class, that class becomes 0. So it is NO LONGER covered.
                # So the prime we just fixed is now safe.
                # But other primes might still be covered.
                # So we continue the loop.
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    best_S = np.zeros(N + 1, dtype=bool)
    best_size = -1
    
    # Deterministic run
    S = np.ones(N + 1, dtype=bool)
    S = _greedy_repair(S.copy(), primes, randomize=False)
    size = np.sum(S)
    if size > best_size:
        best_size = size
        best_S = S.copy()
        
    # Randomized runs if time permits
    # Each run is very fast (ms), so we can do many.
    # Let's do 100 runs.
    for i in range(100):
        if time.time() > deadline:
            break
        S = np.ones(N + 1, dtype=bool)
        S = _greedy_repair(S.copy(), primes, randomize=True)
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    return [int(x) for x in np.where(best_S)[0]]
