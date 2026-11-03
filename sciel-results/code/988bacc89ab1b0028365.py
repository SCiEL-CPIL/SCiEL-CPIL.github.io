"""Construct a finite set U of nonnegative integers containing 0 to maximize the
sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
"""

import time
import random
import math

# Deadline guard: keep a safety margin before the 150 s wall-clock kill.
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _score_set(S):
    """Compute the score for a set S (list of nonnegative ints containing 0)."""
    if not S or 0 not in S:
        return -1.0
    S = sorted(set(S))
    n = len(S)
    maxU = S[-1]
    if maxU == 0:
        return 1.0
    # Compute |U+U| and |U-U|
    # Use set comprehension for correctness; for moderate sizes this is fine.
    sumset = set()
    diffset = set()
    for i, a in enumerate(S):
        for j in range(i, n):
            b = S[j]
            sumset.add(a + b)
        for j in range(n):
            b = S[j]
            diffset.add(a - b)
    # diffset includes negatives; |U-U| is the size of the set of all differences
    # We already computed all a-b for all a,b in S
    # Actually let's recompute properly:
    sumset = set()
    diffset = set()
    for a in S:
        for b in S:
            sumset.add(a + b)
            diffset.add(a - b)
    num_sum = len(sumset)
    num_diff = len(diffset)
    if num_sum == 0:
        return -1.0
    ratio = num_diff / num_sum
    if ratio <= 0:
        return -1.0
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _initial_set(n):
    """Generate an initial set of size n containing 0."""
    if n <= 1:
        return [0]
    # Start with a geometric-like progression
    S = [0]
    # Add elements to spread out
    for i in range(1, n):
        # Use a mix of powers and random
        if i < 6:
            S.append(2 ** i)
        else:
            S.append(random.randint(1, 2 ** (i + 1)))
    # Ensure uniqueness and sorted
    S = sorted(set(S))
    # Pad if needed
    while len(S) < n:
        S.append(random.randint(1, max(S) * 2 + 10))
    S = sorted(set(S))
    return S[:n]

def _local_search(S, max_iters=500, time_budget=None):
    """Perform local search by replacing elements."""
    S = sorted(set(S))
    if 0 not in S:
        S = sorted([0] + S)
    best_score = _score_set(S)
    best_S = list(S)
    
    if time_budget is None:
        time_budget = _remaining() - 5.0  # safety margin
    
    iters = 0
    while iters < max_iters and _remaining() > 5.0:
        iters += 1
        n = len(S)
        # Try replacing a random element (not 0) with a new candidate
        idx = random.randint(1, n - 1)
        old_val = S[idx]
        
        # Generate candidates: neighbors, sums/diffs of existing elements
        candidates = set()
        for x in S:
            for y in S:
                candidates.add(x + y)
                if x >= y:
                    candidates.add(x - y)
        candidates.add(old_val + 1)
        candidates.add(old_val - 1 if old_val > 0 else 1)
        candidates.add(random.randint(0, max(S) * 2 + 50))
        candidates.add(random.randint(0, max(S)))
        
        best_candidate_score = best_score
        best_candidate_val = old_val
        
        for c in list(candidates)[:20]:  # limit candidates tried
            c = max(0, c)
            if c in S and c != old_val:
                continue
            new_S = S[:idx] + [c] + S[idx+1:]
            new_S = sorted(set(new_S))
            if 0 not in new_S:
                new_S = sorted([0] + new_S)
            if len(new_S) != n:
                continue
            sc = _score_set(new_S)
            if sc > best_candidate_score:
                best_candidate_score = sc
                best_candidate_val = c
        
        if best_candidate_val != old_val:
            S[idx] = best_candidate_val
            S = sorted(set(S))
            if best_score < best_candidate_score:
                best_score = best_candidate_score
                best_S = list(S)
    
    return best_S, best_score

def _try_fixed_size(n, time_budget=None):
    """Try to optimize a set of fixed size n."""
    if time_budget is None:
        time_budget = _remaining() - 5.0
    
    if time_budget < 2.0:
        return None, -1.0
    
    # Start with a few random initializations
    best_S = None
    best_score = -1.0
    
    num_starts = max(1, int(time_budget / 10.0))
    num_starts = min(num_starts, 3)
    
    for _ in range(num_starts):
        S = _initial_set(n)
        S, sc = _local_search(S, max_iters=300, time_budget=time_budget - 2.0)
        if sc > best_score:
            best_score = sc
            best_S = list(S)
    
    return best_S, best_score

def construct():
    """Main entry point: construct a set U maximizing the score."""
    if _remaining() < 10.0:
        # Not enough time; return a small known set
        return [0, 1, 3]
    
    best_S = [0, 1, 3]
    best_score = _score_set(best_S)
    
    # Try various sizes
    sizes_to_try = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
    
    # Sort by expected difficulty (smaller first)
    for n in sizes_to_try:
        if _remaining() < 15.0:
            break
        S, sc = _try_fixed_size(n)
        if S is not None and sc > best_score:
            best_score = sc
            best_S = list(S)
    
    # Final polish: try to improve the best set found
    if _remaining() > 20.0 and best_S is not None:
        polished_S, polished_score = _local_search(
            list(best_S), max_iters=200, time_budget=_remaining() - 5.0
        )
        if polished_score > best_score:
            best_score = polished_score
            best_S = list(polished_S)
    
    return best_S
