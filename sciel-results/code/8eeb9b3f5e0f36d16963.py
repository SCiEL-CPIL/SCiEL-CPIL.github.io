import time
import random
import math
from itertools import combinations

def _subset_sums(S):
    """Return set of all nonempty subset sums of S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _is_valid(S):
    """Full validity check for a set S.
    Check 1: All subset sums are distinct (no two subsets have same sum)
    Check 2: No two distinct subset sums divide each other
    """
    if not S:
        return True
    
    # Check for duplicate subset sums using the property:
    # If two subsets have the same sum, their symmetric difference gives
    # two disjoint subsets with the same sum. But easier: just compute all sums
    # and check for duplicates.
    
    # Actually, for a set of distinct positive integers, subset sums can collide.
    # We need to check that all subset sums are distinct.
    # For k elements, there are 2^k - 1 nonempty subsets.
    # If any two have the same sum, invalid.
    
    # Efficient way: use the fact that if sum(A) = sum(B) for A != B,
    # then A\B and B\A are disjoint with equal sum.
    # But let's just compute all sums and check size.
    
    sums = _subset_sums(S)
    # Number of nonempty subsets is 2^len(S) - 1
    expected = (1 << len(S)) - 1
    if len(sums) != expected:
        return False
    
    # Check divisibility among subset sums
    total = sum(S)
    sums_list = sorted(sums)
    sums_set = sums
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    pool = list(range(1, 257))
    
    # Strategy: Use a greedy + local search approach
    # Key insight: larger numbers have fewer multiples, so they're "safer"
    # Also, numbers that are pairwise coprime are safer (less divisibility risk)
    
    # Phase 1: Greedy initialization
    # Sort by: prefer large numbers, prefer numbers with few divisors in range
    
    def divisor_count_in_range(x, lo=1, hi=256):
        """Count how many numbers in [lo,hi] divide x or are divisible by x"""
        count = 0
        for y in range(lo, hi+1):
            if y == x:
                continue
            if x % y == 0 or y % x == 0:
                count += 1
        return count
    
    # Precompute "risk" for each number
    risk = {}
    for x in pool:
        # A number is risky if it divides many others or is divided by many others
        # in the pool. This is a proxy for subset sum divisibility risk.
        divs = 0
        for y in pool:
            if x != y and (x % y == 0 or y % x == 0):
                divs += 1
        risk[x] = divs
    
    # Greedy: start with numbers that have low risk (few divisibility relations)
    # and are large (fewer multiples fit in the sum range)
    candidates = sorted(pool, key=lambda x: (risk[x], -x))
    
    current_set = []
    current_sums = set()
    current_total = 0
    used = set()
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if x in used:
            continue
        # Try adding x
        new_sums = set(current_sums)
        for s in list(current_sums):
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Quick check: do subset sums collide?
        expected = (1 << (len(current_set) + 1)) - 1
        if len(new_sums) != expected:
            continue
        
        # Check divisibility
        new_total = current_total + x
        new_sums_set = new_sums
        valid = True
        for s in new_sums:
            m = 2 * s
            while m <= new_total:
                if m in new_sums_set:
                    valid = False
                    break
                m += s
            if not valid:
                break
        
        if valid:
            current_set.append(x)
            current_sums = new_sums
            current_total = new_total
            used.add(x)
            if len(current_set) >= 30:
                break
    
    best_set = list(current_set)
    best_size = len(current_set)
    
    if best_size >= 30:
        return [int(x) for x in best_set]
    
    # Phase 2: Local search to improve
    # Try swapping elements to find better configurations
    rng = random.Random(42)
    T = 1.0
    T_min = 0.01
    
    current = list(best_set)
    cur_sums = set(best_sums) if (best_sums := _subset_sums(best_set)) else set()
    cur_total = sum(current)
    cur_size = len(current)
    
    steps = 0
    max_steps = 20000
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        steps += 1
        
        if not current:
            break
        
        # Try to add a new element
        improved = False
        for _ in range(3):
            add_elem = rng.choice(pool)
            if add_elem in current:
                continue
            new_sums = set(cur_sums)
            for s in list(cur_sums):
                new_sums.add(s + add_elem)
            new_sums.add(add_elem)
            
            expected = (1 << (cur_size + 1)) - 1
            if len(new_sums) != expected:
                continue
            
            new_total = cur_total + add_elem
            new_sums_set = new_sums
            valid = True
            for s in new_sums:
                m = 2 * s
                while m <= new_total:
                    if m in new_sums_set:
                        valid = False
                        break
                    m += s
                if not valid:
                    break
            
            if valid:
                current = current + [add_elem]
                cur_sums = new_sums
                cur_total = new_total
                cur_size += 1
                improved = True
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = list(current)
                if best_size >= 30:
                    break
                break
        
        if best_size >= 30:
            break
        
        # Try a swap: remove one, add another
        if not improved:
            idx = rng.randint(0, len(current) - 1)
            remove_elem = current[idx]
            temp_set = current[:idx] + current[idx+1:]
            temp_sums = _subset_sums(temp_set)
            temp_total = sum(temp_set)
            
            for _ in range(3):
                add_elem = rng.choice(pool)
                if add_elem in temp_set:
                    continue
                new_sums = set(temp_sums)
                for s in list(temp_sums):
                    new_sums.add(s + add_elem)
                new_sums.add(add_elem)
                
                expected = (1 << len(temp_set)) - 1  # len(temp_set) = cur_size - 1, +1 = cur_size
                if len(new_sums) != expected:
                    continue
                
                new_total = temp_total + add_elem
                new_sums_set = new_sums
                valid = True
                for s in new_sums:
                    m = 2 * s
                    while m <= new_total:
                        if m in new_sums_set:
                            valid = False
                            break
                    m += s
                if not valid:
                    break
                
                if valid:
                    new_set = temp_set + [add_elem]
                    # Same size, accept with some probability to explore
                    if rng.random() < 0.3:
                        current = new_set
                        cur_sums = new_sums
                        cur_total = new_total
                        # cur_size unchanged
        
        T *= 0.999
        if T < T_min:
            T = T_min
    
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback: try to trim until valid
        while best_set and not _is_valid(best_set):
            best_set.pop()
            best_size = len(best_set)
    
    return [int(x) for x in best_set]
