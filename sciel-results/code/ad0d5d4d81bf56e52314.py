"""Optimized program for small maximal Sidon sets in [1,256]."""
import time
import random
import sys

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

def _greedy_sidon_maximal_bitset(seed=None):
    """
    Build a maximal Sidon set using greedy approach with bitset optimization.
    Returns the set.
    """
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    S = []
    sums_mask = 0
    
    for x in candidates:
        # Check if x can be added to S
        # New sums: 2*x, x+a
        if sums_mask & (1 << (2*x)):
            continue
            
        conflict = False
        for a in S:
            if sums_mask & (1 << (x + a)):
                conflict = True
                break
                
        if conflict:
            continue
            
        # Add x
        S.append(x)
        sums_mask |= (1 << (2*x))
        for a in S[:-1]:
            sums_mask |= (1 << (x + a))
            
    S.sort()
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1

    def elapsed():
        return time.time() - start_time

    # Run many random restarts with the fast bitset greedy
    # Since the check is fast, we can run thousands of iterations.
    
    for seed in range(20000):
        if elapsed() > _DEADLINE - 2:
            break
            
        S = _greedy_sidon_maximal_bitset(seed)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
        # Optional: If we find a very small set, we can stop early
        # but it's hard to know the lower bound.
        if best_size <= 10:
            break

    if best_set is None:
        # Fallback
        best_set = _greedy_sidon_maximal_bitset(0)
        
    return best_set
