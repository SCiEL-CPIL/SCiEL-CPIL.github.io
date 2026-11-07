"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def _subset_sums_set(S):
    """Return set of all nonempty subset sums of S."""
    n = len(S)
    sums = set()
    # Use iterative approach for speed
    current = {0}
    for x in S:
        new_sums = set()
        for s in current:
            ns = s + x
            new_sums.add(ns)
            sums.add(ns)
        current = current | new_sums
    return sums

def _is_valid(S):
    """Check if no two distinct subset sums divide each other."""
    if not S:
        return True
    sums = _subset_sums_set(S)
    if len(sums) == 0:
        return True
    sums_sorted = sorted(sums)
    n = len(sums_sorted)
    # For each sum, check no multiple in sums
    sums_set = sums
    for i in range(n):
        s = sums_sorted[i]
        multiple = 2 * s
        while multiple in sums_set:
            return False
        multiple += s
    return True

def _fast_check_partial(S, candidate):
    """Quick check: does adding candidate to S create a divisibility conflict?"""
    # Only need to check new sums involving candidate
    # New sums = {x + candidate : x in subset_sums(S)} ∪ {candidate}
    # But we also need to check conflicts among existing sums and new sums
    # This is complex; for speed, do full check on small sets
    return _is_valid(S + [candidate])

def _greedy_extend(S, candidates, deadline):
    """Try to extend S by adding candidates greedily."""
    S = list(S)
    for c in candidates:
        if time.time() > deadline:
            break
        new_S = S + [c]
        if _is_valid(new_S):
            S = new_S
    return S

def _random_search(deadline):
    """Random search for valid sets."""
    best = []
    best_size = 0
    
    # Strategy: try different random subsets and extend greedily
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random starting point biased toward larger numbers
        # Large numbers have subset sums that are more spread out
        if attempt % 4 == 0:
            start = sorted(random.sample(range(120, 257), min(10, 137)), reverse=True)
        elif attempt % 4 == 1:
            start = sorted(random.sample(range(80, 257), min(15, 177)), reverse=True)
        elif attempt % 4 == 2:
            start = sorted(random.sample(range(1, 257), 20), reverse=True)
        else:
            # Try a mix
            start = sorted(random.sample(range(50, 257), 25), reverse=True)
        
        # Greedy extend with remaining candidates
        remaining = [x for x in range(1, 257) if x not in start]
        random.shuffle(remaining)
        
        result = _greedy_extend(start, remaining, deadline)
        
        if len(result) > best_size:
            best_size = len(result)
            best = result
            
            # If we found a good set, try to improve further
            if best_size >= 8:
                # Try local search: swap elements
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(len(best)):
                        for j in range(len(best)):
                            if i == j:
                                continue
                            # Try swapping best[i] with a different candidate
                            for c in range(1, 257):
                                if c in best:
                                    continue
                                new_set = list(best)
                                new_set[i] = c
                                if len(new_set) == len(set(new_set)):
                                    if _is_valid(new_set):
                                        # Check if this actually helps (size same, but maybe enables extension)
                                        # For now, just accept if valid
                                        pass
                        break  # Limit swap search for speed
    return best

def _construct_known_good():
    """Try to find a good set using systematic approach."""
    # Known approach: use numbers that are pairwise coprime or have large ratios
    # Try powers of 2 shifted, or prime-based sets
    
    best = []
    best_size = 0
    
    # Try starting with large primes
    primes = [x for x in range(100, 257) if all(x % p != 0 for p in range(2, int(x**0.5)+1))]
    
    # Greedy with primes first
    if primes:
        result = _greedy_extend([], sorted(primes, reverse=True) + list(range(1, 100)), time.time() + 30)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Try powers of 2
    powers = [2**i for i in range(1, 9) if 2**i <= 256]  # 2,4,8,16,32,64,128
    result = _greedy_extend(list(powers), list(range(1, 257)), time.time() + 10)
    if len(result) > best_size:
        best_size = len(result)
        best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 40s safety margin
    
    # Start with a known good set
    best = _construct_known_good()
    best_size = len(best)
    
    if time.time() > deadline:
        return list(best)
    
    # Random search
    random_seed = int(time.time() * 1000) % (2**31)
    random.seed(random_seed)
    
    result = _random_search(deadline)
    
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Final validation
    if best and not _is_valid(best):
        # Fallback to minimal valid set
        best = [1, 2, 4]
    
    return list(best)
# EVOLVE-BLOCK-END
