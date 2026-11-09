import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin

    best_set = [31, 32, 34]
    best_size = 3

    def compute_subset_sums(S):
        """Compute all subset sums of S as a list (includes 0)."""
        sums = [0]
        for x in S:
            sums += [s + x for s in sums]
        return sums

    def is_valid_full(S):
        """Full validation of set S."""
        if not S:
            return True
        sums = compute_subset_sums(S)
        non_empty = [s for s in sums if s != 0]
        # Check distinctness
        if len(set(non_empty)) != len(non_empty):
            return False
        # Check divisibility
        n = len(non_empty)
        for i in range(n):
            u = non_empty[i]
            for j in range(n):
                if i == j:
                    continue
                v = non_empty[j]
                if u != v and (v % u == 0 or u % v == 0):
                    return False
        return True

    def can_add(S, x, sums_cache=None):
        """Check if x can be added to valid set S. Uses cached sums if provided."""
        if sums_cache is None:
            sums_cache = compute_subset_sums(S)
        new_sums = [s + x for s in sums_cache]
        # Check distinctness: new_sums should not overlap with sums_cache (except 0)
        if any(ns in sums_cache for ns in new_sums if ns != 0):
            return False
        # Check divisibility among new_sums
        non_empty_new = [s for s in new_sums if s != 0]
        non_empty_old = [s for s in sums_cache if s != 0]
        # Internal check
        for i in range(len(non_empty_new)):
            u = non_empty_new[i]
            for j in range(len(non_empty_new)):
                if i == j:
                    continue
                v = non_empty_new[j]
                if u != v and (v % u == 0 or u % v == 0):
                    return False
        # Cross check
        for u in non_empty_new:
            for v in non_empty_old:
                if u != v and (v % u == 0 or u % v == 0):
                    return False
        return True

    def greedy_build(order):
        """Build a set greedily following the given order."""
        S = []
        sums_cache = [0]
        for x in order:
            if time.time() - start_time > time_limit - 1:
                break
            if can_add(S, x, sums_cache):
                S.append(x)
                sums_cache += [s + x for s in sums_cache]
        return S

    def local_search_2opt(S, all_candidates):
        """Try 2-opt swaps to improve set S."""
        if not S:
            return S
        S = list(S)
        current_sums = compute_subset_sums(S)
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            n = len(S)
            # Try removing one element and adding a different one
            for i in range(n):
                if time.time() - start_time > time_limit - 2:
                    break
                S_without_i = S[:i] + S[i+1:]
                sums_without_i = compute_subset_sums(S_without_i)
                for x in all_candidates:
                    if x in S:
                        continue
                    if can_add(S_without_i, x, sums_without_i):
                        S_new = S_without_i + [x]
                        # Validate
                        if is_valid_full(S_new):
                            S = S_new
                            current_sums = compute_subset_sums(S)
                            improved = True
                            break
        return S

    # Phase 1: Multiple random greedy builds
    all_candidates = list(range(1, 257))
    
    # Try different orderings
    for restart in range(500):
        if time.time() - start_time > time_limit - 5:
            break
        random.shuffle(all_candidates)
        S = greedy_build(all_candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        # Early exit if we're doing very well
        if best_size >= 15:
            break

    # Phase 2: Local search from best set
    if time.time() - start_time < time_limit - 3:
        improved_set = local_search_2opt(best_set, all_candidates)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]

    # Phase 3: More targeted search - try to extend best set
    if time.time() - start_time < time_limit - 1 and best_size < 20:
        S = list(best_set)
        sums_cache = compute_subset_sums(S)
        # Try adding elements in order of "niceness" (larger numbers first)
        for x in sorted(all_candidates, reverse=True):
            if time.time() - start_time > time_limit - 1:
                break
            if x in S:
                continue
            if can_add(S, x, sums_cache):
                S.append(x)
                sums_cache += [s + x for s in sums_cache]
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]

    # Final verification
    if not is_valid_full(best_set):
        best_set = [31, 32, 34]
        best_size = 3

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
