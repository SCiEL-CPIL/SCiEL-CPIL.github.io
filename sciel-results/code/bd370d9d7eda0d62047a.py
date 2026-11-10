"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 5000
_DEADLINE = 320.0

# EVOLVE-BLOCK-START

def _primes_upto(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

PRIMES = _primes_upto(N + 1)

def _greedy_remove(S, rng):
    """
    Greedily remove residue classes to make S admissible.
    S is a list of integers.
    Returns the resulting list.
    """
    S = list(S)
    while True:
        n = len(S)
        violated = None
        for p in PRIMES:
            if p > n:
                break
            residues = set(x % p for x in S)
            if len(residues) == p:
                violated = p
                break
        if violated is None:
            break
        p = violated
        # Count frequencies
        freq = {}
        for x in S:
            r = x % p
            freq[r] = freq.get(r, 0) + 1
        min_freq = min(freq.values())
        candidates = [r for r, c in freq.items() if c == min_freq]
        r_to_remove = rng.choice(candidates)
        S = [x for x in S if x % p != r_to_remove]
    return S

def _recover_incremental(S, removed, primes, rng):
    """
    Try to add back elements from 'removed' to S using incremental residue tracking.
    S is a list, removed is a list of candidates not in S.
    Returns the augmented list.
    """
    S = list(S)
    S_set = set(S)
    removed = list(removed)
    rng.shuffle(removed)
    
    # Precompute covered residues for each relevant prime
    # Only primes up to N matter (since len(S) <= N+1)
    covered = {}
    for p in primes:
        if p > N + 1:
            break
        covered[p] = set(x % p for x in S)
    
    # We'll process removed in batches, trying to add each
    # For efficiency, we check only primes <= len(S)+1 when adding
    added_any = True
    while added_any and removed:
        added_any = False
        new_removed = []
        for x in removed:
            new_n = len(S) + 1
            valid = True
            # Check primes <= new_n
            for p in primes:
                if p > new_n:
                    break
                r = x % p
                if r not in covered[p]:
                    # Adding this residue would complete the set for prime p
                    if len(covered[p]) == p - 1:
                        valid = False
                        break
            if valid:
                S.append(x)
                S_set.add(x)
                # Update covered for primes <= len(S)
                for p in primes:
                    if p > len(S):
                        break
                    covered[p].add(x % p)
                added_any = True
            else:
                new_removed.append(x)
        removed = new_removed
    
    return S

def _greedy_add_incremental(S, candidates, primes, rng):
    """
    Add elements from candidates to S if they maintain admissibility,
    using incremental residue tracking.
    """
    S = list(S)
    S_set = set(S)
    candidates = list(candidates)
    rng.shuffle(candidates)
    
    covered = {}
    for p in primes:
        if p > N + 1:
            break
        covered[p] = set(x % p for x in S)
    
    added_any = True
    while added_any and candidates:
        added_any = False
        new_candidates = []
        for x in candidates:
            new_n = len(S) + 1
            valid = True
            for p in primes:
                if p > new_n:
                    break
                r = x % p
                if r not in covered[p]:
                    if len(covered[p]) == p - 1:
                        valid = False
                        break
            if valid:
                S.append(x)
                S_set.add(x)
                for p in primes:
                    if p > len(S):
                        break
                    covered[p].add(x % p)
                added_any = True
            else:
                new_candidates.append(x)
        candidates = new_candidates
    
    return S

def _local_search_swaps(S, complement, primes, rng, max_rounds=50):
    """
    Try 1-out-2-in swaps: remove x from S, try to add y, z from complement.
    If S - {x} + {y, z} is admissible, accept.
    """
    S = list(S)
    S_set = set(S)
    comp = list(complement)
    rng.shuffle(comp)
    
    # Precompute covered residues
    covered = {}
    for p in primes:
        if p > N + 1:
            break
        covered[p] = set(x % p for x in S)
    
    best = S
    best_size = len(S)
    
    for _ in range(max_rounds):
        improved = False
        # For each element in S, try removing it and adding two from comp
        # This is expensive, so we limit the search
        S_sample = rng.sample(S, min(len(S), 100))
        for x in S_sample:
            # Remove x
            S_temp = [v for v in S if v != x]
            S_temp_set = S_set - {x}
            
            # Update covered temporarily
            covered_temp = {}
            for p in primes:
                if p > N + 1:
                    break
                covered_temp[p] = set(x2 % p for x2 in S_temp)
            
            # Try to add two elements from comp
            added = []
            comp_remaining = list(comp)
            rng.shuffle(comp_remaining)
            
            for y in comp_remaining:
                if y in S_temp_set:
                    continue
                new_n = len(S_temp) + 1
                valid = True
                for p in primes:
                    if p > new_n:
                        break
                    r = y % p
                    if r not in covered_temp[p]:
                        if len(covered_temp[p]) == p - 1:
                            valid = False
                            break
                if valid:
                    S_temp.append(y)
                    S_temp_set.add(y)
                    for p in primes:
                        if p > len(S_temp):
                            break
                        covered_temp[p].add(y % p)
                    added.append(y)
                    break
            
            if len(added) == 1:
                # Try to add one more
                z_added = None
                for z in comp_remaining:
                    if z in S_temp_set or z == added[0]:
                        continue
                    new_n = len(S_temp) + 1
                    valid = True
                    for p in primes:
                        if p > new_n:
                            break
                        r = z % p
                        if r not in covered_temp[p]:
                            if len(covered_temp[p]) == p - 1:
                                valid = False
                                break
                    if valid:
                        S_temp.append(z)
                        S_temp_set.add(z)
                        for p in primes:
                            if p > len(S_temp):
                                break
                            covered_temp[p].add(z % p)
                        z_added = z
                        break
            
            if len(added) == 2 or (len(added) == 1 and z_added is not None):
                # Check if we actually improved size
                if len(S_temp) > best_size:
                    S = S_temp
                    S_set = S_temp_set
                    best = S
                    best_size = len(S)
                    # Update complement
                    comp = [v for v in comp if v not in S_set]
                    # Recompute covered
                    covered = {}
                    for p in primes:
                        if p > N + 1:
                            break
                        covered[p] = set(x2 % p for x2 in S)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy removals with different random tie-breaking
    rng = random.Random(42)
    
    # First, do a deterministic greedy to get a baseline
    S_init = list(range(N + 1))
    S_greedy = _greedy_remove(S_init, random.Random(0))
    best = S_greedy
    best_size = len(S_greedy)
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 60:
        return best
    
    # Phase 2: Recovery from the greedy result
    S_set = set(S_greedy)
    removed = [x for x in range(N + 1) if x not in S_set]
    S_recovered = _recover_incremental(S_greedy, removed, PRIMES, random.Random(1))
    if len(S_recovered) > best_size:
        best = S_recovered
        best_size = len(S_recovered)
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 40:
        return best
    
    # Phase 3: More restarts with different random seeds for greedy
    for seed in range(10):
        if time.time() - start_time > _DEADLINE - 20:
            break
        rng = random.Random(seed + 100)
        S_greedy = _greedy_remove(S_init, rng)
        S_set = set(S_greedy)
        removed = [x for x in range(N + 1) if x not in S_set]
        S_recovered = _recover_incremental(S_greedy, removed, PRIMES, random.Random(seed + 200))
        if len(S_recovered) > best_size:
            best = S_recovered
            best_size = len(S_recovered)
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 15:
        return best
    
    # Phase 4: Local search swaps on the best solution
    if best_size > 0:
        best_set = set(best)
        complement = [x for x in range(N + 1) if x not in best_set]
        S_swapped = _local_search_swaps(best, complement, PRIMES, random.Random(999), max_rounds=30)
        if len(S_swapped) > best_size:
            best = S_swapped
            best_size = len(S_swapped)
    
    return best

# EVOLVE-BLOCK-END
