"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))

    # Precompute reciprocal as integer: L // i
    recip = [L // i for i in range(1, 23)]

    # Try to find maximum cardinality set via backtracking
    # Order: try smaller indices first (larger reciprocals) for better pruning
    order = list(range(1, 23))

    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    def check_and_add(current_sums, r):
        """Check if adding r to current subset sums creates any collision.
        Returns new set of sums if valid, None if collision."""
        new_sums = set()
        for s in current_sums:
            ns = s + r
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        current_sums.update(new_sums)
        return current_sums

    def backtrack(selected, remaining, current_sums):
        """Recursively try to add elements from remaining to selected."""
        nonlocal best_result
        if time.time() > deadline:
            return
        if len(selected) > len(best_result):
            best_result = selected[:]

        if not remaining:
            return

        for i in range(len(remaining)):
            idx = remaining[i]
            r = recip[idx - 1]
            # Try adding idx
            new_sums = set(current_sums)
            valid = True
            for s in list(current_sums):
                ns = s + r
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                selected.append(idx)
                backtrack(selected, remaining[i+1:], new_sums)
                selected.pop()
            # If we've already found a set of size 16, we can stop trying smaller
            if len(best_result) >= 16:
                return

    # Start backtracking
    backtrack([], order, {0})

    return best_result
# EVOLVE-BLOCK-END
