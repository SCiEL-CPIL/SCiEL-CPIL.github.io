import numpy as np
import time
import random
from collections import defaultdict

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _greedy_construct(seed, deadline):
    random.seed(seed)
    
    # We want to maximize the size of the set S.
    # Constraint: For all primes p <= |S|, S does not cover all residues mod p.
    # Strategy: Iteratively remove residue classes for primes.
    # For a given prime p, if S covers all residues, we MUST remove at least one.
    # To maximize |S|, we remove the residue class with the fewest elements.
    
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    # We need to process primes in increasing order? 
    # Actually, the condition is p <= |S|. 
    # If we start with |S| = N, we need to satisfy constraints for all p <= N.
    # As we remove elements, |S| drops. 
    # It is generally safe to process primes in increasing order because removing elements
    # for smaller primes reduces the counts for larger primes, but we still need to ensure
    # that for the current |S|, no prime p <= |S| has full coverage.
    
    # However, a subtle point: if we remove elements to satisfy p=2, the set size drops.
    # Then we check p=3. If p=3 > new |S|, we don't need to satisfy p=3?
    # No, the condition is "for every prime p <= len(tuple)". 
    # So if len(tuple) becomes 10, we only need to check primes 2, 3, 5, 7.
    # We do NOT need to check 11, 13, etc.
    
    # So the algorithm:
    # 1. Start with S = {0..N}.
    # 2. For p in PRIMES:
    #    a. If p > current_size: break (no more primes to check).
    #    b. Check if S covers all residues mod p.
    #       - If not, continue.
    #       - If yes, we MUST remove at least one residue class.
    #       - Find residue r with min count in S.
    #       - Remove all x in S where x % p == r.
    #       - Update current_size.
    # 3. Return S.
    
    # Optimization: 
    # Checking "covers all residues" is O(p) if we have counts.
    # We can maintain counts for each prime? That's expensive memory-wise if we do it for all primes.
    # But N=5000, primes up to 5000 are ~669.
    # We can just compute counts on the fly using NumPy bincount for each prime.
    # O(N) per prime. 669 * 5000 = 3.3M ops. This is very fast in NumPy.
    
    for p in PRIMES:
        if time.time() > deadline:
            break
        if p > current_size:
            break
            
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        counts = np.bincount(residues, minlength=p)
        
        # If any count is 0, we are already admissible for this p.
        if np.any(counts == 0):
            continue
            
        # If all counts > 0, we must remove a class.
        # Remove the class with the MINIMUM count to preserve maximum size.
        min_count = np.min(counts)
        
        # Random tie-breaking among min counts
        candidate_classes = np.where(counts == min_count)[0]
        r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        mask = (x_indices % p == r) & active
        removed = np.sum(mask)
        current_size -= removed
        active = active & ~mask
        
        if current_size == 0:
            break
            
    return np.where(active)[0].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Multi-start greedy
    attempts = 0
    # We can run many attempts because the greedy is fast.
    # Let's estimate: ~1ms per attempt? 290s / 1ms = 290,000 attempts.
    # Even if it's 10ms, 29,000 attempts.
    
    while time.time() < deadline - 2:
        attempts += 1
        seed = attempts
        points = _greedy_construct(seed, deadline)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
    return best_points
