"""Construct the largest admissible tuple in [0, 5000]."""
import time
import random
import math

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

# Precompute primes up to N (and beyond for checking)
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

PRIMES = _primes(N + 1)

def _get_smallest_violated_prime(S, primes):
    """
    Find the smallest prime p <= len(S) for which all residues mod p are covered.
    Returns the prime or None if S is admissible.
    """
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _greedy_remove(S, primes, rng=None):
    """
    Greedily remove residue classes to make S admissible.
    If rng is provided, tie-breaking is random; otherwise deterministic (smallest class).
    """
    S = list(S)
    while True:
        p = _get_smallest_violated_prime(S, primes)
        if p is None:
            break
        
        # Count frequencies of each residue mod p
        freq = {}
        for x in S:
            r = x % p
            freq[r] = freq.get(r, 0) + 1
        
        # Find min freq
        min_freq = min(freq.values())
        # Get all classes with min freq
        candidates = [r for r, c in freq.items() if c == min_freq]
        
        # Pick one to remove
        if rng:
            r_to_remove = rng.choice(candidates)
        else:
            # Deterministic: pick smallest residue for consistency
            r_to_remove = min(candidates)
        
        # Remove all elements with this residue
        S = [x for x in S if x % p != r_to_remove]
        
    return S

def _recover(S, removed, primes):
    """
    Try to add elements back from 'removed' to S, maintaining admissibility.
    This is the key refinement step.
    """
    S = list(S)
    removed = list(removed)
    
    # Precompute covered residues for each prime
    # We only need primes up to N, but we check dynamically
    # Optimization: Only track primes that are currently <= len(S) or might become relevant
    # But for simplicity and correctness, let's track all primes up to N.
    # Actually, we can just re-check on the fly or maintain a dict.
    # Let's maintain a dict of sets for efficiency.
    
    # Initialize covered sets
    covered = {}
    for p in primes:
        if p > N:
            break
        covered[p] = set(x % p for x in S)
        
    added_any = True
    # Sort removed to try smaller numbers first? Or random?
    # Let's try a mix. Smaller numbers might be "easier" to fit?
    # Actually, order doesn't matter for the final set size, just which set we get.
    # Let's iterate through removed.
    
    while added_any and removed:
        added_any = False
        new_removed = []
        
        # Iterate through removed elements
        # To make it faster, we can shuffle or just iterate.
        # Let's iterate.
        for x in removed:
            new_n = len(S) + 1
            valid = True
            
            # Check if adding x violates admissibility
            # We need to check all primes p <= new_n
            for p in primes:
                if p > new_n:
                    break
                r = x % p
                # If r is not in covered[p], adding x adds a new residue.
                # This is bad if covered[p] is already full (size p-1).
                if r not in covered[p]:
                    if len(covered[p]) == p - 1:
                        valid = False
                        break
            
            if valid:
                # Add x
                S.append(x)
                # Update covered sets
                for p in primes:
                    if p > len(S):
                        break
                    covered[p].add(x % p)
                added_any = True
            else:
                new_removed.append(x)
                
        removed = new_removed
        
    return S

def _greedy_construct():
    """
    Start with all integers and greedily remove to make admissible.
    """
    points = list(range(N + 1))
    return _greedy_remove(points, PRIMES, rng=None)

def _randomized_greedy(seed=0):
    """
    Randomized variant: start with a random subset and greedily repair.
    """
    rng = random.Random(seed)
    # Start with a random subset of size ~ N * 0.7
    target_size = int(N * 0.7)
    points = rng.sample(range(N + 1), target_size)
    
    return _greedy_remove(points, PRIMES, rng=rng)

def construct():
    """
    Construct the largest admissible tuple in [0, 5000].
    Returns a list of distinct integers.
    """
    start_time = time.time()
    
    # 1. Deterministic Greedy
    best = _greedy_construct()
    best_len = len(best)
    
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME - 30:
        return best

    # 2. Try randomized greedy + recovery
    # The recovery step is expensive, so we limit the number of restarts.
    # We will alternate between simple greedy (fast) and greedy+recovery (slower).
    
    best_found = best
    best_len = len(best)
    
    seed = 0
    while time.time() - start_time < MAX_TIME - 10:
        # Check time before starting a new iteration
        if time.time() - start_time > MAX_TIME - 5:
            break
            
        # Alternate strategy
        if seed % 2 == 0:
            # Randomized Greedy + Recovery
            candidate_greedy = _randomized_greedy(seed=seed)
            # Determine removed elements for recovery
            candidate_set = set(candidate_greedy)
            removed = [x for x in range(N + 1) if x not in candidate_set]
            
            candidate = _recover(candidate_greedy, removed, PRIMES)
        else:
            # Just Randomized Greedy (faster, allows more restarts)
            candidate = _randomized_greedy(seed=seed)
            
        if len(candidate) > best_len:
            best_len = len(candidate)
            best_found = candidate
            
        seed += 1
        
    return best_found

# EVOLVE-BLOCK-END
