"""Rewritten program: Optimized greedy and local search for divisibility-incomparable subset sums."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _is_valid(nums):
    """Check if all nonempty subset sums are pairwise non-dividing."""
    n = len(nums)
    if n == 0:
        return True
    
    # Generate all nonempty subset sums
    sums = set()
    for x in nums:
        new_sums = {s + x for s in sums}
        new_sums.add(x)
        sums = sums | new_sums
    
    if len(sums) < 2:
        return True
    
    # Sort sums for efficient divisibility check
    sorted_sums = sorted(sums)
    n_sums = len(sorted_sums)
    
    # For each sum, check if any larger sum is divisible by it
    for i in range(n_sums):
        a = sorted_sums[i]
        # Check multiples of a in the set
        for j in range(i + 1, n_sums):
            b = sorted_sums[j]
            if b > a * 2:  # b cannot be a multiple of a if b > 2a (since all sums are positive)
                break
            if b % a == 0:
                return False
    
    return True

def _subset_sums(nums):
    """Generate all nonempty subset sums."""
    sums = set()
    for x in nums:
        new_sums = {s + x for s in sums}
        new_sums.add(x)
        sums = sums | new_sums
    return sums

def _greedy_build(seed, deadline, start_size=4):
    """Build a valid set greedily, starting from a small valid set."""
    rng = random.Random(seed)
    
    # Start with a known valid small set
    base_sets = [
        [2, 3, 5, 7],      # primes, likely valid
        [3, 5, 7, 11],     # larger primes
        [4, 7, 11, 13],    # mixed
    ]
    
    # Try to find a valid starting set of size start_size
    current = []
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    
    # Build up to start_size
    for _ in range(start_size):
        added = False
        for x in candidates:
            if time.time() > deadline:
                return current
            test = current + [x]
            if _is_valid(test):
                current.append(x)
                candidates.remove(x)
                added = True
                break
        if not added:
            break
    
    if len(current) < 2:
        return [2, 3]  # Fallback
    
    # Now try to add more elements
    sums = _subset_sums(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates for diversity
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            test = current + [x]
            if _is_valid(test):
                current.append(x)
                candidates.remove(x)
                sums = _subset_sums(current)
                improved = True
                break
        
        if not improved and len(current) > 2:
            # Try a swap: remove one element, try to add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                temp = current[:i] + current[i+1:]
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    test = temp + [x]
                    if _is_valid(test) and len(test) > len(temp):
                        current = test
                        candidates.remove(x)
                        candidates.append(removed)
                        improved = True
                        break
                if improved:
                    break
    
    return current

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    best = [2, 3, 5, 7]
    best_len = 4
    
    # Try multiple seeds with different strategies
    seeds = range(1, 200)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        cur = _greedy_build(seed, deadline, start_size=4)
        
        if len(cur) > best_len:
            # Verify
            if _is_valid(cur):
                best = cur
                best_len = len(cur)
                
                # If we found a large set, try to improve it
                if best_len >= 10:
                    # Do a local search on the best set
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        cur_sums = _subset_sums(best)
                        candidates = [x for x in range(1, 257) if x not in best]
                        random.shuffle(candidates)
                        
                        for x in candidates:
                            if time.time() > deadline:
                                break
                            test = best + [x]
                            if _is_valid(test):
                                best = test
                                improved = True
                                break
                        
                        if not improved and len(best) > 5:
                            # Try swaps
                            for i in range(len(best)):
                                if time.time() > deadline:
                                    break
                                removed = best[i]
                                temp = best[:i] + best[i+1:]
                                
                                for x in candidates:
                                    if time.time() > deadline:
                                        break
                                    test = temp + [x]
                                    if _is_valid(test) and len(test) > len(temp):
                                        best = test
                                        candidates.remove(x)
                                        candidates.append(removed)
                                        improved = True
                                        break
                                if improved:
                                    break
    
    return list(best)

# EVOLVE-BLOCK-END
