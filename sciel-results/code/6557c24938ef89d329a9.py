"""Rewritten program: meet-in-the-middle search for minimum terms."""
import time
from collections import defaultdict

def _divisors_of(n):
    """Return all positive divisors of n, sorted descending."""
    out = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
    return sorted(out, reverse=True)

def construct():
    deadline = time.time() + 50
    n = 5040
    target = 4979
    divs = _divisors_of(n)
    divs = [d for d in divs if d <= target]
    m = len(divs)
    
    # Split divisors into two halves for meet-in-the-middle
    half = m // 2
    left = divs[:half]
    right = divs[half:]
    
    # For each half, precompute: for each achievable sum, the minimum number of terms
    # and one such combination (stored as a tuple of indices into the half)
    # Use BFS: start with sum=0, count=0, indices=()
    # Expand by adding one element at a time, tracking (sum, count, indices)
    # For each sum, keep the minimum count; if tied, keep the first found
    
    def precompute_half(arr):
        """Return dict: sum -> (min_count, tuple_of_indices)"""
        best = {0: (0, ())}
        # BFS by count level
        current_level = {0: (0, ())}
        for level in range(1, len(arr) + 1):
            next_level = {}
            for s, (cnt, idx) in current_level.items():
                for i in range(len(arr)):
                    if i in idx:
                        continue
                    new_s = s + arr[i]
                    if new_s > target:
                        continue
                    if new_s not in next_level or level < next_level[new_s][0]:
                        next_level[new_s] = (level, idx + (i,))
            for s, val in next_level.items():
                if s not in best or val[0] < best[s][0]:
                    best[s] = val
            current_level = next_level
            if time.time() > deadline:
                break
        return best
    
    left_sums = precompute_half(left)
    right_sums = precompute_half(right)
    
    # Combine: find min total count such that s1 + s2 = target
    # Iterate over all s1 in left_sums, check if target - s1 in right_sums
    best_total = None
    best_combo = None
    
    for s1, (c1, idx1) in left_sums.items():
        s2 = target - s1
        if s2 in right_sums:
            c2, idx2 = right_sums[s2]
            total = c1 + c2
            if best_total is None or total < best_total:
                best_total = total
                best_combo = (idx1, idx2)
                if total == 1:  # can't do better
                    break
        if time.time() > deadline:
            break
    
    if best_combo is not None:
        idx1, idx2 = best_combo
        result = []
        for i in idx1:
            result.append(left[i])
        for i in idx2:
            result.append(right[i])
        return result
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
