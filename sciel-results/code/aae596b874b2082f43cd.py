import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # We want distinct integers in 1..256.
    # No two distinct subset sums may divide one another.
    # Equal sums from different subsets are invalid.
    # Maximize set size.

    # Key structural insight:
    # If all elements are > 128, then:
    #   - Any single element is in (128, 256]
    #   - Any sum of 2+ elements is in (256, 512]
    #   - For a single element x to divide a 2-element sum y+z (where y,z > 128):
    #     y+z >= 2*129 = 258, and x <= 256
    #     So x could divide y+z only if y+z = 2x (since y+z < 2*256 = 512 < 3x for x > 128)
    #     Wait, 3x > 384, and y+z < 512, so x could divide y+z if y+z = 2x.
    #   - For a single element x to divide a 3+ element sum: sum >= 3*129 = 387, and x <= 256
    #     x could divide the sum if sum = 2x or 3x. But 3x > 384, and sum of 3 elements from (128,256] is in [387, 768].
    #     So x could divide sum if sum = 2x or 3x.
    #   
    # This is complex. Let me use a different approach:
    # 
    # Use the "powers of 2" idea but with a twist. If we pick numbers that are all 
    # in a range where their subset sums don't create divisibility, we can get large sets.
    #
    # Actually, the best approach for this type of problem is to use numbers that are
    # all close to each other and large, so that subset sums are all in a narrow range
    # where no one divides another.
    #
    # Let's try: pick numbers from [128, 256]. The subset sums range from 128 to 256*k.
    # For k elements, max sum is 256k, min sum is 128k.
    # For two sums a and b to have a|b, we need b >= 2a (since a != b and b > a).
    # If all elements are in [m, 2m], then:
    #   - Single element sums: in [m, 2m]
    #   - Two element sums: in [2m+1, 4m-2] (approximately)
    #   - A single element x in [m, 2m] divides a two-element sum s in [2m+1, 4m-2]
    #     only if s = 2x (since 3x > 3m+3 > 4m-2 for m >= 4, wait let me check)
    #     3x >= 3m+3, and 4m-2 < 3m+3 when m < 5. For m >= 5, 3m+3 > 4m-2.
    #     So for m >= 5, a single element can only divide a two-element sum if s = 2x.
    #
    # Let me just use a computational approach with efficient validation.

    def is_valid_set(S):
        """Check if set S satisfies all conditions."""
        if len(S) == 0:
            return True
        if len(S) != len(set(S)):
            return False
        if any(x < 1 or x > 256 for x in S):
            return False
        
        elements = list(S)
        n = len(elements)
        sums_set = set()
        
        # Use iterative subset sum computation
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums_set:
                return False
            sums_set.add(s)
        
        # Check divisibility
        sums_list = sorted(sums_set)
        n_sums = len(sums_list)
        for i in range(n_sums):
            for j in range(i + 1, n_sums):
                if sums_list[j] % sums_list[i] == 0:
                    return False
        return True

    def can_add(S, candidate):
        """Check if candidate can be added to S without violating conditions."""
        if candidate in S:
            return False
        if candidate < 1 or candidate > 256:
            return False
        
        elements = list(S)
        n = len(elements)
        current_sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            current_sums.add(s)
        
        new_sums = set()
        new_sums.add(candidate)
        for s in current_sums:
            new_sums.add(s + candidate)
        
        all_sums = current_sums | new_sums
        if len(all_sums) != len(current_sums) + len(new_sums):
            return False
        
        for a in new_sums:
            for b in all_sums:
                if a == b:
                    continue
                if a % b == 0 or b % a == 0:
                    return False
        return True

    def greedy_construct(order, deadline):
        S = set()
        for c in order:
            if time.time() > deadline:
                break
            if can_add(S, c):
                S.add(c)
        return S

    def local_search(S, candidates, deadline):
        best_S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_c in list(best_S):
                if time.time() > deadline:
                    break
                temp_S = best_S - {remove_c}
                for add_c in candidates:
                    if add_c in temp_S:
                        continue
                    if can_add(temp_S, add_c):
                        new_S = temp_S | {add_c}
                        if len(new_S) > len(best_S):
                            best_S = new_S
                            improved = True
                            break
            if not improved:
                for add_c in candidates:
                    if add_c in best_S:
                        continue
                    if can_add(best_S, add_c):
                        best_S.add(add_c)
                        improved = True
        return best_S

    # Strategy: Try multiple different approaches
    best_S = set()
    best_size = 0

    # Approach 1: Greedy from large to small
    candidates_large = list(range(256, 127, -1))
    S1 = greedy_construct(candidates_large, deadline)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)

    # Approach 2: Randomized greedy with different seeds
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        shuffled = list(range(3, 257))
        random.shuffle(shuffled)
        S_trial = greedy_construct(shuffled, deadline)
        if len(S_trial) > best_size:
            best_S = S_trial
            best_size = len(S_trial)

    # Approach 3: Local search to improve
    if time.time() < deadline - 5:
        candidates_all = list(range(3, 257))
        improved_S = local_search(best_S, candidates_all, deadline - 2)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(best_S)

    # Final verification
    if is_valid_set(best_S):
        return [int(x) for x in sorted(best_S)]
    
    return [3]
