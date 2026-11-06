"""B_3[1] set construction: maximize cardinality with unique unordered triple sums.

Structural change: use a NumPy-based representation for the set of "used" triple
sums, and a different search strategy — simulated annealing / local search on the
complement (which elements to remove) starting from a dense greedy set, combined
with a multi-pass greedy with varied candidate orderings.
"""
import time
import random
import numpy as np
from itertools import combinations_with_replacement

DEADLINE = 85.0  # safety margin before 90s limit


def _all_triple_sums(S):
    """Compute all unordered triple sums (with repetition) for set S."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums


def _triple_sums_with_new(x, S):
    """Compute all triple sums that involve x, given existing sorted list S."""
    sums = set()
    n = len(S)
    sums.add(3 * x)
    for s in S:
        sums.add(2 * x + s)
        sums.add(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.add(x + S[i] + S[j])
    return sums


def _greedy_with_order(candidates):
    """Greedy build using given candidate order."""
    S = []
    sum_set = set()
    for x in candidates:
        new_sums = _triple_sums_with_new(x, S)
        if not new_sums & sum_set:
            S.append(x)
            sum_set.update(new_sums)
    return S, sum_set


def _greedy_linear():
    """Linear greedy from 1 to 4096."""
    return _greedy_with_order(range(1, 4097))


def _greedy_exponential(base):
    """Exponential spacing greedy."""
    candidates = []
    x = 1
    while x <= 4096:
        candidates.append(x)
        x = x * base + 1
    # Then fill in remaining in ascending order
    candidates = sorted(set(candidates) | set(range(1, 4097)))
    return _greedy_with_order(candidates)


def _random_greedy(seed_val):
    """Random order greedy."""
    rng = random.Random(seed_val)
    candidates = list(range(1, 4097))
    rng.shuffle(candidates)
    return _greedy_with_order(candidates)


def _greedy_multi_step():
    """Multi-phase greedy with decreasing step sizes."""
    S = []
    sum_set = set()
    for step in [64, 32, 16, 8, 4, 2, 1]:
        for x in range(1, 4097, step):
            if x in sum_set:  # quick skip if x itself is used
                continue
            new_sums = _triple_sums_with_new(x, S)
            if not new_sums & sum_set:
                S.append(x)
                sum_set.update(new_sums)
    return S, sum_set


def _local_search_remove(S, sum_set, max_time):
    """Try to improve by removing elements that allow adding others.
    
    Strategy: find pairs (remove r, add a) where removing r frees up space
    for adding a (and possibly more).
    """
    start = time.time()
    S = list(S)
    sum_set = set(sum_set)
    n = len(S)
    S_set = set(S)
    
    # Precompute which elements are "tight" (removing them would help)
    # For each element r in S, compute what sums would be freed
    # Then check if any candidate x not in S could be added
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        # Try each element in S as a removal candidate
        for idx, r in enumerate(S):
            # Compute sums involving r
            r_sums = _triple_sums_with_new(r, [x for i, x in enumerate(S) if i != idx])
            # Try to add each candidate x not in S
            for x in range(1, 4097):
                if x in S_set:
                    continue
                # If we remove r and add x, we need:
                # 1. All sums involving x (with S - {r}) not in (sum_set - r_sums)
                # 2. This is expensive, so only try a few candidates
                # Quick check: x + 2*min(S without r) should not be in sum_set - r_sums
                S_without_r = [x for i, x in enumerate(S) if i != idx]
                if not S_without_r:
                    break
                new_sums_x = _triple_sums_with_new(x, S_without_r)
                freed = sum_set - r_sums
                if not (new_sums_x & freed):
                    # We can remove r and add x
                    # Also check if we can add more
                    S_new = [x for i, x in enumerate(S) if i != idx] + [x]
                    S_new.sort()
                    S_set_new = S_set - {r} | {x}
                    sum_set_new = freed | new_sums_x
                    # Try to add more elements
                    for x2 in range(1, 4097):
                        if x2 in S_set_new:
                            continue
                        new_sums_x2 = _triple_sums_with_new(x2, S_new)
                        if not (new_sums_x2 & sum_set_new):
                            S_new.append(x2)
                            S_new.sort()
                            S_set_new.add(x2)
                            sum_set_new.update(new_sums_x2)
                    if len(S_new) > len(S):
                        S = S_new
                        S_set = S_set_new
                        sum_set = sum_set_new
                        improved = True
                        break
                if (time.time() - start) > max_time:
                    break
            if (time.time() - start) > max_time:
                break
    return S, sum_set


def _numpy_greedy():
    """Use numpy for fast sum tracking. Represent used sums as a boolean array."""
    max_sum = 3 * 4096  # maximum possible triple sum
    used = np.zeros(max_sum + 1, dtype=bool)
    S = []
    
    for x in range(1, 4097):
        # Compute all triple sums involving x
        # 3*x, 2*x+s, x+2*s, x+s1+s2 for s in S
        s_arr = np.array(S, dtype=np.int64) if S else np.array([], dtype=np.int64)
        
        new_sums = set()
        new_sums.add(3 * x)
        if len(S) > 0:
            new_sums.update((2 * x + s_arr).tolist())
            new_sums.update((x + 2 * s_arr).tolist())
            # x + s_i + s_j for i < j
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    new_sums.add(x + S[i] + S[j])
        
        # Check for conflicts using numpy
        arr = np.array(sorted(new_sums), dtype=np.int64)
        if len(arr) > 0:
            if used[arr].any():
                continue
        
        # Accept
        S.append(x)
        used[arr] = True
    
    return S


def _verify(S):
    """Verify all unordered triples have unique sums."""
    if len(S) < 3:
        return True
    sums = [sum(t) for t in combinations_with_replacement(sorted(S), 3)]
    return len(sums) == len(set(sums))


def construct():
    start_time = time.time()
    
    best = []
    
    # Strategy 1: Linear greedy (baseline)
    S1, _ = _greedy_linear()
    if len(S1) > len(best):
        best = S1
    if time.time() - start_time > DEADLINE - 10:
        return best
    
    # Strategy 2: Multi-step greedy
    S2, _ = _greedy_multi_step()
    if len(S2) > len(best):
        best = S2
    if time.time() - start_time > DEADLINE - 8:
        return best
    
    # Strategy 3: Numpy-accelerated greedy
    S3 = _numpy_greedy()
    if len(S3) > len(best):
        best = S3
    if time.time() - start_time > DEADLINE - 6:
        return best
    
    # Strategy 4: Random greedy with multiple seeds
    for seed_val in [1, 7, 42, 123, 999, 2024, 31415, 271828, 12345, 54321]:
        S4 = _random_greedy(seed_val)
        if len(S4) > len(best):
            best = S4
        if time.time() - start_time > DEADLINE - 3:
            break
    
    # Strategy 5: Local search on the best set found so far
    if best and (time.time() - start_time) < DEADLINE - 5:
        S_best, sum_set_best = _greedy_with_order(sorted(best))
        max_ls_time = min(3.0, DEADLINE - 5 - (time.time() - start_time))
        if max_ls_time > 0.5:
            S_ls, _ = _local_search_remove(S_best, sum_set_best, max_ls_time)
            if len(S_ls) > len(best):
                best = S_ls
    
    # Final verification
    if best and not _verify(best):
        # Fallback to verified smaller set
        S_fallback, _ = _greedy_linear()
        if _verify(S_fallback):
            best = S_fallback
    
    return best


if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Elements: {result[:20]}...")
    print(f"Valid: {_verify(result)}")
