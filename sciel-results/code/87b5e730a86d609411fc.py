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

    # Order elements by descending reciprocal (larger values first for better pruning)
    order = sorted(range(1, 23), key=lambda x: -recip[x-1])

    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_result)

    def backtrack(selected, remaining_idx, sums):
        """Recursively try to add elements. sums is a sorted list of all subset sums."""
        nonlocal best_result, best_size
        if time.time() > deadline:
            return
        if len(selected) > best_size:
            best_result = selected[:]
            best_size = len(selected)

        # Pruning: can we possibly beat best_size?
        if len(selected) + len(remaining_idx) <= best_size:
            return

        for i in range(len(remaining_idx)):
            idx = remaining_idx[i]
            r = recip[idx - 1]

            # Check collision: for each existing sum s, s+r must not be in sums
            # Use two-pointer or set-based check
            # For efficiency, check if any s+r collides with another s'+r or existing sum
            new_sums = []
            valid = True
            # sums is sorted; s+r for all s in sums gives a sorted list
            # We need: all (s+r) are distinct from each other AND from all elements in sums
            # Since sums is sorted, s+r is also sorted
            # Check if any s+r equals any element in sums, or if any s+r equals any other s'+r
            # s+r == s'+r iff s==s', so the s+r values are always distinct from each other
            # So we only need to check if any s+r is in sums
            # Use a set for O(1) lookup
            sums_set = set(sums)
            for s in sums:
                ns = s + r
                if ns in sums_set:
                    valid = False
                    break

            if valid:
                # Build new sums list: merge sums and sums+r (both sorted)
                new_sums_list = []
                j = 0
                k = 0
                len_sums = len(sums)
                while j < len_sums or k < len_sums:
                    if k >= len_sums or (j < len_sums and sums[j] < sums[k] + r):
                        new_sums_list.append(sums[j])
                        j += 1
                    elif j >= len_sums or sums[k] + r < sums[j]:
                        new_sums_list.append(sums[k] + r)
                        k += 1
                    else:
                        # equal - but we already checked no collision, so this shouldn't happen
                        new_sums_list.append(sums[j])
                        new_sums_list.append(sums[k] + r)
                        j += 1
                        k += 1

                selected.append(idx)
                backtrack(selected, remaining_idx[i+1:], new_sums_list)
                selected.pop()

                if best_size >= 16:
                    return

    # Start backtracking
    backtrack([], order, [0])

    return best_result
# EVOLVE-BLOCK-END
