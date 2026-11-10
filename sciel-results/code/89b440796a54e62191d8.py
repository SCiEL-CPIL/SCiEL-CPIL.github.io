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

def _is_admissible(S, primes):
    """Check if S is admissible: for every prime p <= len(S), some residue class mod p is missed."""
    size = np.sum(S)
    indices = np.where(S)[0]
    if len(indices) == 0:
        return True
    for p in primes:
        if p > size:
            break
        residues = indices % p
        counts = np.bincount(residues, minlength=p)
        if np.all(counts > 0):
            return False
    return True

def _greedy_repair(S, primes, randomize=False, rng=None):
    """
    Repairs S to be admissible by removing elements.
    S is a boolean array of length N+1.
    Returns the modified S (boolean array).
    """
    if rng is None:
        rng = random
    
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
                min_classes = np.where(counts == min_count)[0]
                
                if randomize and len(min_classes) > 1:
                    r_to_remove = min_classes[rng.randint(0, len(min_classes) - 1)]
                else:
                    r_to_remove = min_classes[0]
                
                mask = (residues == r_to_remove)
                S[indices[mask]] = False
                changed = True
                
    return S

def _local_search_improve(S, primes, time_budget, randomize=True, rng=None):
    """
    Attempts to improve S by trying to add back elements that were removed.
    This is a local search where we try to add elements one by one and repair if needed.
    """
    if rng is None:
        rng = random
        
    S = S.copy()
    start_time = time.time()
    
    # Find all positions not in S
    candidate_positions = np.where(~S)[0]
    if len(candidate_positions) == 0:
        return S
    
    # Shuffle candidates
    rng.shuffle(candidate_positions)
    
    current_size = np.sum(S)
    
    for pos in candidate_positions:
        if time.time() - start_time > time_budget:
            break
            
        # Try adding pos
        S_test = S.copy()
        S_test[pos] = True
        new_size = np.sum(S_test)
        
        # Check if still admissible
        is_adm = True
        for p in primes:
            if p > new_size:
                break
            indices = np.where(S_test)[0]
            if len(indices) == 0:
                break
            residues = indices % p
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                is_adm = False
                break
        
        if is_adm:
            S = S_test
            current_size = new_size
        else:
            # Try to repair by removing some elements
            S_repaired = _greedy_repair(S_test, primes, randomize=randomize, rng=rng)
            if np.sum(S_repaired) > current_size:
                S = S_repaired
                current_size = np.sum(S)
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    best_S = np.zeros(N + 1, dtype=bool)
    best_size = -1
    
    rng = random.Random(42)
    
    # Deterministic run
    S = np.ones(N + 1, dtype=bool)
    S = _greedy_repair(S.copy(), primes, randomize=False)
    size = np.sum(S)
    if size > best_size:
        best_size = size
        best_S = S.copy()
        
    # Randomized repair runs
    for i in range(200):
        if time.time() > deadline - 30:
            break
        S = np.ones(N + 1, dtype=bool)
        S = _greedy_repair(S.copy(), primes, randomize=True, rng=rng)
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
    # Local search improvement on the best solution
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        # Split time: 70% for local search, 30% for more randomized runs
        local_search_time = remaining_time * 0.7
        
        improved_S = _local_search_improve(best_S, primes, local_search_time, randomize=True, rng=rng)
        improved_size = np.sum(improved_S)
        if improved_size > best_size:
            best_size = improved_size
            best_S = improved_S.copy()
            
    # If there's still time, do more randomized runs
    if time.time() < deadline - 2:
        for i in range(50):
            if time.time() > deadline - 2:
                break
            S = np.ones(N + 1, dtype=bool)
            S = _greedy_repair(S.copy(), primes, randomize=True, rng=rng)
            size = np.sum(S)
            if size > best_size:
                best_size = size
                best_S = S.copy()
                
    return [int(x) for x in np.where(best_S)[0]]
