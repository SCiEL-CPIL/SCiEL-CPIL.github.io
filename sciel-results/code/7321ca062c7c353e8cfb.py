import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def is_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct."""
        k = len(elements)
        if k <= 1:
            return True
        
        # Compute LCM
        L = reduce(lcm, elements)
        
        # Convert to integers
        int_vals = [L // x for x in elements]
        
        # Quick check: Superincreasing property
        # If the largest element is greater than the sum of all others,
        # all subset sums are distinct.
        int_vals.sort()
        if int_vals[-1] > sum(int_vals[:-1]):
            return True
        
        # Full check: Generate all subset sums
        # Use a set to track sums
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
            
        return len(sums) == (1 << k)

    def greedy_fill(current_set):
        """
        Takes a set of elements, and greedily adds more elements from 1..22
        in random order, as long as dissociation is maintained.
        """
        current = list(current_set)
        remaining = [x for x in range(1, 23) if x not in current_set]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            test_set = current + [x]
            if is_dissociated(test_set):
                current = test_set
        return current

    # Initial best set: Powers of 2 are always dissociated
    # 1, 2, 4, 8, 16
    best_set = [1, 2, 4, 8, 16]
    best_size = len(best_set)
    
    # Also try a set of primes as another seed
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    if is_dissociated(primes):
        if len(primes) > best_size:
            best_set = primes
            best_size = len(primes)
            
    # Try to improve from empty set with random greedy
    while time.time() < deadline - 5:
        # 1. Random Greedy from scratch
        candidate = greedy_fill(set())
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)
            
        # 2. Local Search: Remove one from best, re-fill
        if best_size > 0 and time.time() < deadline - 5:
            # Pick a random element to remove
            y = random.choice(best_set)
            base_set = set(best_set) - {y}
            candidate = greedy_fill(base_set)
            if len(candidate) > best_size:
                best_set = candidate
                best_size = len(candidate)
                
        # 3. Random Swap: Replace one element in best with another
        if best_size > 0 and time.time() < deadline - 5:
            y = random.choice(best_set)
            # Pick a random x not in best_set
            candidates_for_swap = [x for x in range(1, 23) if x not in best_set]
            if candidates_for_swap:
                x = random.choice(candidates_for_swap)
                base_set = set(best_set) - {y} | {x}
                candidate = greedy_fill(base_set)
                if len(candidate) > best_size:
                    best_set = candidate
                    best_size = len(candidate)

    return best_set
