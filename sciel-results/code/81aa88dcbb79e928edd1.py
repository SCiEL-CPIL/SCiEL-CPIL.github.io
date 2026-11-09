"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
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

# Precompute primes up to N
_all_primes = _primes(N)

def _construct_tuple(residues, primes_to_use, k):
    """
    Given residues for primes_to_use, compute the set of survivors in [0, N].
    Return the first k survivors as a list.
    """
    indices = np.arange(N + 1)
    alive = np.ones(N + 1, dtype=bool)
    
    for p, r in zip(primes_to_use, residues):
        mask = (indices % p) == r
        alive[mask] = False
        
    result_indices = np.where(alive)[0]
    
    if len(result_indices) < k:
        return []
        
    return result_indices[:k].tolist()

def _is_admissible(tup, k):
    """
    Check if a tuple of size k is admissible.
    For every prime p <= k, some residue class mod p is missed.
    """
    if len(tup) < k:
        return False
        
    primes_to_use = [p for p in _all_primes if p <= k]
    
    for p in primes_to_use:
        residues_present = set()
        for x in tup:
            residues_present.add(x % p)
        if len(residues_present) == p:
            return False
            
    return True

def _simulate_knockout(k, time_budget):
    """
    Structural change: Simulate a knockout tournament among candidate tuples.
    Instead of binary searching on k and local search on residues,
    we generate diverse candidate tuples via randomized greedy + local search,
    and let them 'compete' by checking if they can be extended or if they
    are valid for a larger k.
    
    Actually, a more direct structural change is to use a **Randomized Restarts**
    strategy with a **Different Objective**: 
    Maximize the size of the set directly by adding elements one by one,
    ensuring admissibility is maintained. This is a constructive greedy approach.
    
    Let's try: **Greedy Construction with Backtracking**.
    Start with an empty set.
    At each step, try to add a new element x in [0, N] such that the set remains
    'potentially admissible' for the new size k+1.
    
    Condition for admissibility of size k:
    For all primes p <= k, not all residues mod p are present.
    
    When we add x, the set size becomes k+1.
    We must ensure that for all primes p <= k+1, not all residues are present.
    For p <= k, the condition was already met. Adding x might fill the last missing
    residue for some p <= k, which would break admissibility.
    So we must check: for each p <= k, if (residues_present(p) | {x%p}) == all residues, skip x.
    For p = k+1 (if prime), we must ensure that not all residues are present.
    Initially, for a new prime p, we have k residues. Since k < p (usually), we can't
    cover all p residues. So the condition for p=k+1 is automatically satisfied as long as
    k < k+1, which is true. Wait, if k+1 is prime, we have k elements. Can k elements cover
    all k+1 residues mod k+1? No, because k < k+1. So the condition for the new prime
    is automatically satisfied.
    
    So the constraint is ONLY on the primes p <= k (the old primes).
    We must not complete the residue set for any p <= k.
    
    Algorithm:
    1. Start with empty set S.
    2. While we can add more elements:
       a. Determine the next size k+1.
       b. Find candidates x in [0, N] not in S.
       c. Filter candidates: x is valid if for all p <= k, (residues(S, p) ∪ {x%p}) != {0, ..., p-1}.
       d. If no valid candidates, stop.
       e. Pick one valid candidate (e.g., random, or one that minimizes future risk).
       f. Add to S.
    
    This is a very different approach: **Constructive Greedy**.
    It avoids the complex residue optimization and binary search.
    It directly builds the tuple.
    
    To improve, we can use a **Lookahead** or **Backtracking** if we get stuck.
    Or simply run multiple restarts with different random seeds.
    
    Let's implement this.
    """
    start_time = time.time()
    S = set()
    residues = {p: set() for p in _all_primes} # residues[p] = set of residues present in S mod p
    
    # We need to check primes up to N eventually, but we only care about p <= len(S).
    # So we can maintain a list of active primes.
    
    # To speed up, we can precompute for each x, which primes p <= current_k are 'critical'
    # i.e., adding x would complete the residue set mod p.
    
    # Let's just do a simple loop.
    # For efficiency, we can maintain a 'blocked' set of x values that are currently invalid.
    # An x is blocked if for some p <= k, residues(S, p) ∪ {x%p} == all residues.
    # This means x%p is the ONLY missing residue for p.
    
    # Let missing_residues[p] = {0, ..., p-1} - residues(S, p).
    # If len(missing_residues[p]) == 1, let r be that residue.
    # Then any x with x%p == r is blocked.
    
    # So we can maintain a set of blocked x values.
    # When we add a new x, we update residues and missing_residues.
    # This might unblock some x values? No, adding x only reduces missing_residues.
    # So the set of blocked x values only grows.
    
    # Wait, if we add x, residues(S, p) increases. missing_residues[p] decreases.
    # If missing_residues[p] was {r1, r2}, and x%p == r1, then it becomes {r2}.
    # The blocked x values for p change from {x: x%p in {r1, r2}} to {x: x%p == r2}.
    # So the set of blocked x values for p SHRINKS.
    # But the union of blocked x values over all p might grow or shrink.
    
    # This dynamic update is complex. Let's just re-evaluate candidates each step.
    # N=5000 is small. k will be at most ~100-200.
    # So the loop runs ~200 times.
    # Each step, we check up to 5000 candidates.
    # For each candidate, we check up to ~20 primes.
    # 200 * 5000 * 20 = 20,000,000 operations. This is fine in Python with NumPy?
    # Maybe a bit slow in pure Python. Let's use NumPy for the check.
    
    # Actually, let's try a simpler version first and see the speed.
    
    candidates = list(range(N + 1))
    random.shuffle(candidates)
    
    while True:
        if time.time() > start_time + time_budget:
            break
            
        k = len(S)
        if k > N:
            break
            
        # Find a valid candidate
        valid_candidate = None
        for x in candidates:
            if x in S:
                continue
                
            # Check if x is valid
            is_valid = True
            for p in _all_primes:
                if p > k:
                    break
                    
                # If adding x completes the residue set mod p, it's invalid
                if (x % p) not in residues[p]:
                    # x introduces a new residue
                    # Check if this was the last missing residue
                    if len(residues[p]) == p - 1:
                        # It was missing exactly one, and x provides it
                        is_valid = False
                        break
            
            if is_valid:
                valid_candidate = x
                break
                
        if valid_candidate is None:
            break
            
        # Add valid_candidate to S
        S.add(valid_candidate)
        for p in _all_primes:
            if p > len(S):
                break
            residues[p].add(valid_candidate % p)
            
    return list(S)

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    
    Strategy:
    1. Use a Constructive Greedy approach with random restarts.
    2. Each restart builds a tuple by adding elements one by one, ensuring
       admissibility is maintained.
    3. Keep the best tuple found.
    """
    
    best_tuple = []
    best_size = 0
    
    # Run multiple restarts with different time budgets
    # We have ~300s total.
    # Let's do 10 restarts, each with ~25s budget?
    # Or adaptive: run restarts until time is up.
    
    start_time = time.time()
    restart = 0
    
    while time.time() < _DEADLINE - 10.0:
        restart += 1
        # Allocate time for this restart
        # If we have lots of time, give it more.
        # Let's say 20s per restart, but adjust if we are near deadline.
        remaining = _DEADLINE - 10.0 - time.time()
        if remaining < 5.0:
            break
            
        time_budget = min(20.0, remaining - 5.0)
        
        # Set random seed for reproducibility within a run, but vary across restarts
        random.seed(12345 + restart)
        
        tup = _simulate_knockout(0, time_budget)
        
        # Verify admissibility
        k = len(tup)
        if _is_admissible(tup, k):
            if k > best_size:
                best_size = k
                best_tuple = tup
        else:
            # Should not happen, but if it does, trim the tuple
            # Find the largest prefix that is admissible
            for i in range(len(tup), 0, -1):
                sub_tup = tup[:i]
                if _is_admissible(sub_tup, i):
                    best_tuple = sub_tup
                    best_size = i
                    break
                    
    return best_tuple

# EVOLVE-BLOCK-END
