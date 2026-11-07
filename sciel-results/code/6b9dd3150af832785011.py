"""Cardinality-separated subset sums: maximize set size via greedy + local search."""

import time
import random
import sys

sys.setrecursionlimit(10000)


def _subset_sum_sets(arr, max_k):
    """For each k in 0..max_k, return the set of all subset sums of exactly k elements."""
    n = len(arr)
    # sums[k] = set of all sums of k-element subsets
    sums = [set() for _ in range(max_k + 1)]
    sums[0].add(0)
    for v in arr:
        # iterate in reverse to avoid using v more than once
        for k in range(min(max_k, n), 0, -1):
            prev = sums[k - 1]
            cur = sums[k]
            if not prev:
                continue
            # add v to each sum in prev
            new_sums = set()
            for s in prev:
                ns = s + v
                new_sums.add(ns)
            cur.update(new_sums)
    return sums


def _is_valid(arr, max_k=None):
    """Check if the sorted array arr satisfies the cardinality-separated condition.
    For all k >= 1: all sums of k-element subsets must be disjoint from
    all sums of (k-1)-element subsets (for k-1 >= 0).
    Equivalent: for each k, min_k_sum > max_{k-1}_sum is NOT required (repeats within
    same cardinality are allowed), but no sum in cardinality k can equal a sum in
    cardinality k-1 (or any other different cardinality).
    
    Actually re-reading: "Whenever two subsets have different cardinalities their sums must differ."
    This means: for any k1 != k2, the set of all k1-subset-sums must be disjoint from
    the set of all k2-subset-sums.
    """
    n = len(arr)
    if n == 0:
        return True
    if max_k is None:
        max_k = n
    
    # Compute all subset sums by cardinality
    sums = [set() for _ in range(max_k + 1)]
    sums[0].add(0)
    for v in arr:
        for k in range(min(max_k, n), 0, -1):
            if sums[k - 1]:
                sums[k].update(s + v for s in sums[k - 1])
    
    # Check disjointness between all pairs of different cardinalities
    for k1 in range(max_k + 1):
        for k2 in range(k1 + 1, max_k + 1):
            if sums[k1] and sums[k2] and sums[k1] & sums[k2]:
                return False
    return True


def _try_add(arr, candidate, max_k=None):
    """Try adding candidate to arr; return True if valid."""
    if candidate in arr:
        return False
    new_arr = arr + [candidate]
    new_arr.sort()
    n = len(new_arr)
    if max_k is None:
        max_k = n
    return _is_valid(new_arr, max_k)


def _greedy_grow(arr, candidates):
    """Greedy: add candidates in order if valid."""
    result = list(arr)
    for c in candidates:
        if c in result:
            continue
        if _try_add(result, c):
            result.append(c)
            result.sort()
    return result


def _local_search(arr, candidates, n_swaps, random_seed=42):
    """1-for-1 swap local search followed by greedy regrowth."""
    rng = random.Random(random_seed)
    current = list(arr)
    current.sort()
    current_set = set(current)
    
    for _ in range(n_swaps):
        improved = False
        # Try replacing one element with another
        for i, old in enumerate(current):
            for new_c in candidates:
                if new_c in current_set:
                    continue
                # Remove old, add new
                trial = [x for x in current if x != old] + [new_c]
                trial.sort()
                n = len(trial)
                if _is_valid(trial, n):
                    current = trial
                    current_set = set(trial)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Greedy regrowth: try to add more elements
    current = _greedy_grow(current, candidates)
    return current


def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    """
    deadline = time.time() + 55  # safety margin
    
    # The key insight: for a set sorted s[0] < s[1] < ... < s[n-1],
    # we need for each k >= 1:
    #   max sum of (k-1) elements < min sum of k elements
    #   (or more generally, all (k-1)-sums disjoint from all k-sums)
    #
    # The tightest constraint is typically:
    #   sum of k smallest > sum of (k-1) largest
    # i.e., sum(s[0:k]) > sum(s[n-(k-1):n])
    #
    # For consecutive integers starting at c: s[i] = c + i
    # sum(s[0:k]) = k*c + k(k-1)/2
    # sum(s[n-k+1:n]) = (k-1)*c + sum from c+n-k+1 to c+n-1 of (c+j)
    #   = (k-1)*c + (k-1)*(2*c + 2*n - k) / 2
    #   = (k-1)*c + (k-1)*(c + n - k/2)
    #   = (k-1)*(2*c + n - k/2)
    #
    # So we need: k*c + k(k-1)/2 > (k-1)*(2*c + n - k/2)
    #
    # For n=31, c=226: let's verify
    # k=1: 226 > 0 (trivially true for k=1 vs k=0)
    # k=2: 2*226 + 1 = 453 > 1*(2*226 + 31 - 1) = 482? No! 453 < 482.
    
    # Wait, let me re-derive. The condition is that NO k-sum equals any (k-1)-sum.
    # The most restrictive case is when the min k-sum equals the max (k-1)-sum.
    # min k-sum = sum of k smallest = sum(s[0:k])
    # max (k-1)-sum = sum of (k-1) largest = sum(s[n-k+1:n])
    # We need: min k-sum > max (k-1)-sum for all k >= 1
    # (Since all k-sums >= min k-sum and all (k-1)-sums <= max (k-1)-sum,
    #  if min k-sum > max (k-1)-sum then all k-sums > all (k-1)-sums)
    
    # For consecutive c, c+1, ..., c+n-1:
    # min k-sum = k*c + k(k-1)/2
    # max (k-1)-sum = sum of largest (k-1) = sum from c+n-(k-1) to c+n-1
    #   = (k-1)/2 * (2*(c+n-(k-1)) + (k-2))
    #   = (k-1)/2 * (2c + 2n - 2k + 2 + k - 2)
    #   = (k-1)/2 * (2c + 2n - k)
    #   = (k-1)*(c + n - k/2)
    
    # Need: k*c + k(k-1)/2 > (k-1)*(c + n - k/2)
    
    # Let's just use the DP-based check. For small sets (up to ~20), this is fast enough.
    # For larger sets, we use the necessary condition as a filter.
    
    N = 256
    
    # Start with a reasonable base: try consecutive integers
    # For n consecutive starting at c, the binding constraint is at some k.
    # Let's just try to build greedily and use local search.
    
    # Greedy from high values down: start with high values, then try to add lower ones
    # Actually, let's try multiple starting points and strategies.
    
    best = []
    
    # Strategy 1: Consecutive high numbers
    for start in range(N, 0, -1):
        n = N - start + 1
        if n <= len(best):
            break
        arr = list(range(start, N + 1))
        if _is_valid(arr, n):
            best = arr
            break
    
    # Strategy 2: Greedy add from a base
    # Start with some high numbers, then greedily add lower ones
    base_sizes = [10, 15, 20]
    for base_n in base_sizes:
        if base_n <= len(best):
            continue
        # Try different bases
        for offset in range(0, 50, 5):
            base_start = N - base_n + 1 + offset
            if base_start < 1:
                continue
            base = list(range(base_start, base_start + base_n))
            if base[-1] > N:
                continue
            if not _is_valid(base, base_n):
                continue
            
            # Greedily add smaller numbers
            candidates = list(range(1, base_start))
            candidates.sort(reverse=True)  # try larger first? or smaller?
            
            result = _greedy_grow(base, candidates)
            if len(result) > len(best):
                best = result
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 3: Local search from current best
    if len(best) > 0 and time.time() < deadline:
        candidates = [i for i in range(1, N + 1) if i not in best]
        improved = _local_search(best, candidates, n_swaps=200, random_seed=42)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 4: Try to grow best further
    if time.time() < deadline:
        candidates = [i for i in range(1, N + 1) if i not in best]
        candidates.sort(reverse=True)
        best = _greedy_grow(best, candidates)
    
    # Strategy 5: More local search with different seeds
    if time.time() < deadline and len(best) > 0:
        candidates = [i for i in range(1, N + 1) if i not in best]
        for seed in [123, 456, 789]:
            if time.time() > deadline:
                break
            improved = _local_search(best, candidates, n_swaps=100, random_seed=seed)
            if len(improved) > len(best):
                best = improved
            # Try regrowth after local search
            if len(best) < 31:
                remaining = [i for i in range(1, N + 1) if i not in best]
                remaining.sort(reverse=True)
                grown = _greedy_grow(best, remaining)
                if len(grown) > len(best):
                    best = grown
    
    # Final validation
    best.sort()
    n = len(best)
    if n > 0 and not _is_valid(best, n):
        # Fallback to a known valid small set
        best = list(range(128, 136))
    
    return list(best)


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(result)
    # Verify
    assert len(result) == len(set(result)), "Not distinct"
    assert all(1 <= x <= 256 for x in result), "Out of range"
    assert all(isinstance(x, int) for x in result), "Not all ints"
    assert _is_valid(result, len(result)), "Invalid"
