"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
TIME_LIMIT = 290.0  # safety margin before 300s

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_residues_numpy(primes_list, active_mask, N):
    """Choose residue for each prime to minimize marginal removals using numpy."""
    residues = {}
    for p in primes_list:
        # Count active elements in each residue class
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            residues[p] = 0
            continue
        
        counts = np.zeros(p, dtype=np.int32)
        for r in range(p):
            counts[r] = np.sum(active_indices % p == r)
        
        # Choose residue with minimum count
        min_count = np.min(counts)
        best_r = int(np.argmin(counts))
        residues[p] = best_r
        
        # Remove all active x with x % p == best_r
        to_remove = (np.arange(N + 1) % p) == best_r
        active_mask[to_remove] = False
    return residues

def _is_admissible_fast(S, K):
    """Check if S is admissible - for every prime p <= len(S), some residue class mod p is missed."""
    K_actual = len(S)
    if K_actual == 0:
        return True
    primes = _primes(K_actual)
    S_arr = np.array(S)
    for p in primes:
        residues_present = np.unique(S_arr % p)
        if len(residues_present) == p:
            return False
    return True

def _local_search_add(S, removed, K, max_iter=10000):
    """Try to re-add removed elements while maintaining admissibility."""
    S_set = set(S)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        # Try to add each removed element
        for x in range(N + 1):
            if removed[x]:
                new_size = len(S_set) + 1
                if new_size > K:
                    continue
                new_S = list(S_set | {x})
                if _is_admissible_fast(new_S, K):
                    S_set.add(x)
                    removed[x] = False
                    improved = True
                    break
    return list(S_set)

def construct():
    start_time = time.time()
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    best_S = None
    best_size = 0
    
    # Strategy: Try different K values with multiple initializations
    # K determines which primes constrain admissibility
    K_values = [400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
    
    for K in K_values:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        primes_to_use = [p for p in all_primes if p <= K]
        
        # Multiple initialization strategies
        num_trials = 3
        for trial in range(num_trials):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            random.seed(trial * 12345 + K + 7)
            
            active_mask = np.ones(N + 1, dtype=bool)
            
            # Different prime orderings
            primes_order = primes_to_use[:]
            if trial == 0:
                # Smallest first
                primes_order.sort()
            elif trial == 1:
                # Largest first
                primes_order.sort(reverse=True)
            else:
                # Random shuffle
                random.shuffle(primes_order)
            
            residues = _greedy_residues_numpy(primes_order, active_mask, N)
            
            # Compute S
            S = [x for x in range(N + 1) if active_mask[x]]
            size = len(S)
            
            # Check admissibility
            if size <= K and _is_admissible_fast(S, K):
                if size > best_size:
                    best_size = size
                    best_S = S[:]
                
                # Local search: try to add back removed elements
                if time.time() - start_time < TIME_LIMIT - 5:
                    removed = [not active_mask[x] for x in range(N + 1)]
                    improved_S = _local_search_add(S, removed, K, max_iter=5000)
                    if len(improved_S) > best_size:
                        best_size = len(improved_S)
                        best_S = improved_S[:]
    
    # Fallback: use seed algorithm if nothing found
    if best_S is None:
        points = [1] + [p for p in all_primes if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points)):
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best_S = points
    
    return best_S
