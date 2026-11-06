import time
from math import lcm
from functools import reduce
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (5s buffer)

    # LCM of 1..22 for exact integer representation (for verification)
    L = reduce(lcm, range(1, 23))
    
    # Precompute reciprocal as integer: L // i
    # v[i-1] corresponds to element i
    v_exact = [L // i for i in range(1, 23)]
    
    # Modular arithmetic setup for fast greedy search
    P = 10**9 + 7
    # Modular inverse of i modulo P
    # i^{-1} mod P
    v_mod = [pow(i, P - 2, P) for i in range(1, 23)]
    
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_result)
    
    # Helper to verify a candidate set using exact integers
    def is_valid_exact(candidate):
        if not candidate:
            return False
        sums = {0}
        for idx in candidate:
            val = v_exact[idx - 1]
            new_sums = {s + val for s in sums}
            if not sums.isdisjoint(new_sums):
                return False
            sums.update(new_sums)
        return True

    # Verify the initial fallback
    if not is_valid_exact(best_result):
        # Find a small valid set if fallback fails (unlikely)
        best_result = []
        for i in range(1, 23):
            if is_valid_exact(best_result + [i]):
                best_result.append(i)
        best_size = len(best_result)

    indices = list(range(1, 23))
    
    # Greedy Randomized Search with Modular Arithmetic
    # We iterate many times, shuffling the order each time.
    # We track the best valid set found.
    
    while time.time() < deadline:
        random.shuffle(indices)
        sums_mod = {0}
        selected = []
        
        for idx in indices:
            val_mod = v_mod[idx - 1]
            # Generate new modular sums
            new_sums_mod = {s + val_mod for s in sums_mod}
            
            # Check for collision in modular domain
            # If no collision, it's *likely* no collision in integer domain.
            # We verify at the end.
            if sums_mod.isdisjoint(new_sums_mod):
                sums_mod.update(new_sums_mod)
                selected.append(idx)
        
        if len(selected) > best_size:
            # Potential improvement. Verify with exact integers.
            if is_valid_exact(selected):
                best_size = len(selected)
                best_result = selected[:]
                # If we found a very large set, we can't get better than 22.
                if best_size == 22:
                    break
            else:
                # Modular collision didn't happen, but integer collision did.
                # This is rare but possible. We discard this candidate.
                pass
        elif len(selected) == best_size:
            # Same size, no need to update unless we want to try another permutation?
            # Not necessary for maximizing cardinality.
            pass

    return best_result
