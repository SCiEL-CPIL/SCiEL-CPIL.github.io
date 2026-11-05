import random
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    # Precompute weights
    # LCM of 1..22
    D = 1
    for i in range(1, 23):
        D = lcm(D, i)
        
    weights = [D // i for i in range(1, 23)]
    # weights[i-1] corresponds to element i
    
    def is_valid_subset(indices):
        """
        indices: list of indices (0-based) into weights
        Returns True if all subset sums of weights[i] for i in indices are distinct.
        """
        if not indices:
            return True
        k = len(indices)
        # Generate all subset sums
        # Use a set to store sums
        sums = set()
        # Iterate over all 2^k subsets
        # For efficiency, we can use bit masks
        for mask in range(1 << k):
            s = 0
            for j in range(k):
                if mask & (1 << j):
                    s += weights[indices[j]]
            if s in sums:
                return False
            sums.add(s)
        return True

    # We want to find the max size
    # Try sizes from 22 down to 1
    # But we know 10 works. Let's try 12, 11, 10.
    
    best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18] # Seed (0-based: 1,2,3,5,7,11,13,17,19)
    best_size = len(best_set)
    
    for target_size in range(22, best_size, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        # Try random greedy construction
        while time.time() < deadline and not found:
            attempts += 1
            # Random permutation of all 22 elements
            perm = list(range(22))
            random.shuffle(perm)
            
            current_set = []
            for idx in perm:
                if len(current_set) == target_size:
                    break
                # Try adding idx
                new_set = current_set + [idx]
                if is_valid_subset(new_set):
                    current_set = new_set
            
            if len(current_set) == target_size:
                found = True
                best_set = current_set
                best_size = target_size
                # We found a set of this size. Can we find larger?
                # The loop continues to larger sizes? No, we iterate down.
                # So if we found size 12, we break and return?
                # Wait, we iterate from 22 down.
                # So if we find 12, we know 12 is possible.
                # Do we need to check if 13 is possible?
                # The loop structure `for target_size in range(22, ...)` means we try 22 first.
                # If 22 fails, we try 21, etc.
                # So if we find 12, we return it?
                # No, we should check if we can find a larger one first.
                # But we are iterating downwards.
                # So we should iterate UPWARDS?
                # No, we want the MAX.
                # So we should try 22, then 21, etc.
                # If we find 12, it means 22..13 failed.
                # So 12 is the max found so far.
                # We can return it.
                break
        
        if found:
            # Convert 0-based indices to 1-based values
            return [i + 1 for i in best_set]
            
    # If no larger set found, return the seed
    return [i + 1 for i in best_set]
