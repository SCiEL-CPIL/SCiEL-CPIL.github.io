import time
import random
import sys
from math import gcd

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 50  # Safe margin before 90s kill

    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            sums = sums | {s + x for s in sums}
        return sums

    def check_divisibility_fast(sums_set):
        """
        Check if any two distinct sums in sums_set divide each other.
        For each u in sums_set, check if 2u, 3u, ... are in sums_set.
        """
        if len(sums_set) <= 1:
            return True
        max_val = max(sums_set)
        for u in sums_set:
            if u == 0:
                continue
            # Check multiples of u
            m = 2 * u
            while m <= max_val:
                if m in sums_set:
                    return False
                m += u
        return True

    def can_add(x, current_sums):
        """
        Check if we can add x to a set with given subset sums.
        current_sums: set of all subset sums including 0
        Returns (bool, new_sums) if valid.
        """
        new_sums = {x + s for s in current_sums}
        
        # Check distinctness: new_sums should not overlap with current_sums
        if new_sums & current_sums:
            return False, None
        
        # Check divisibility in the combined set
        combined = current_sums | new_sums
        if not check_divisibility_fast(combined - {0}):
            return False, None
        
        return True, combined

    def greedy_build(candidate_order, start_set=None, start_sums=None):
        """Greedy build using a given order of candidates, optionally starting from an existing set."""
        if start_set is not None:
            S = list(start_set)
            current_sums = set(start_sums)
        else:
            S = []
            current_sums = {0}
        
        existing = set(S)
        for x in candidate_order:
            if time.time() - start_time > time_limit:
                break
            if x in existing:
                continue
            valid, new_sums = can_add(x, current_sums)
            if valid:
                S.append(x)
                existing.add(x)
                current_sums = new_sums
        return S, current_sums

    def local_search(S, sums_set):
        """Try to improve the set by removing one element and adding a different one."""
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_copy = list(S)
            sums_copy = set(sums_set)
            
            for remove_idx in range(len(S_copy)):
                if time.time() - start_time > time_limit:
                    break
                
                x_removed = S_copy[remove_idx]
                S_no = S_copy[:remove_idx] + S_copy[remove_idx+1:]
                sums_no = compute_sums(S_no)
                in_set = set(S_no)
                
                # Try adding each candidate not in S_no
                # Prioritize larger numbers to avoid small sum divisibility
                for x in range(256, 0, -1):
                    if x in in_set:
                        continue
                    valid, new_sums = can_add(x, sums_no)
                    if valid:
                        S_new = S_no + [x]
                        if len(S_new) > len(S_copy):
                            S_copy = S_new
                            sums_copy = new_sums
                            improved = True
                            break
                if improved:
                    break
        
        return S_copy, sums_copy

    def score_candidate(x):
        """
        Score a candidate: prefer numbers with fewer small divisors
        (less likely to create divisibility conflicts with subset sums).
        """
        # Count how many values in 1..256 divide x or are multiples of x
        divisors = 0
        for d in range(1, x + 1):
            if x % d == 0:
                divisors += 1
        # Also consider how many multiples x has up to 256
        multiples = 256 // x
        # Prefer numbers with fewer divisors and fewer multiples
        return -(divisors + multiples)

    best_set = []
    best_size = 0
    best_sums = {0}
    
    random.seed(12345)
    
    # Precompute a good ordering: sort by score (ascending score = better candidates first)
    all_candidates = list(range(1, 257))
    scored = [(score_candidate(x), x) for x in all_candidates]
    scored.sort()
    scored_order = [x for _, x in scored]
    
    # Strategy 1: Greedy with scored ordering (prefer numbers with fewer divisors/multiples)
    S, sums = greedy_build(scored_order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
        best_sums = sums

    # Strategy 2: Greedy with large numbers first
    candidates_large = list(range(256, 0, -1))
    S, sums = greedy_build(candidates_large)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
        best_sums = sums

    # Strategy 3: Random shuffles
    num_random = 30
    for i in range(num_random):
        if time.time() - start_time > time_limit - 10:
            break
        candidates = all_candidates[:]
        random.shuffle(candidates)
        S, sums = greedy_build(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_sums = sums

    # Strategy 4: Start with best, try to extend with scored order
    if best_set:
        candidates = [x for x in scored_order if x not in set(best_set)]
        S, sums = greedy_build(candidates, best_set, best_sums)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_sums = sums

    # Strategy 5: Local Search to improve best
    if best_size >= 2 and time.time() - start_time < time_limit - 5:
        S_ls, sums_ls = local_search(best_set, best_sums)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = S_ls[:]
            best_sums = sums_ls

    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        # Check distinctness (should be guaranteed by can_add, but verify)
        if len(non_empty) != (1 << len(best_set)) - 1:
            best_set = []
        elif not check_divisibility_fast(non_empty):
            best_set = []

    if not best_set or len(best_set) < 1:
        return [256]
    
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(res)
