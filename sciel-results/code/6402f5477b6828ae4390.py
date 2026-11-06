"""Rewritten program: pick a dense ambient set whose largest Sidon subset is small."""
import time
import itertools

_DEADLINE = 90.0
_T0 = time.time()


def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _T0)  # 15s safety margin


def _is_sidon(arr):
    """Check if all unordered pair sums (including diagonals) are unique."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _max_sidon_bruteforce(arr):
    """Exact largest Sidon subset via checking subsets from largest to smallest."""
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            if _is_sidon([arr[i] for i in subset]):
                return k
    return 0


def _max_sidon_smart(arr):
    """
    Faster exact largest Sidon subset using backtracking with pruning.
    Uses the fact that for small n (12), we can be smart about ordering and pruning.
    """
    n = len(arr)
    best = [0]
    
    # Precompute all pair sums for quick conflict checking
    # We'll use a set of sums for the current subset
    
    def backtrack(remaining, current_subset, current_sums):
        if not remaining:
            if len(current_subset) > best[0]:
                best[0] = len(current_subset)
            return
        
        # Upper bound: even if we take all remaining, can we beat best?
        if len(current_subset) + len(remaining) <= best[0]:
            return
        
        # Try each element in remaining
        for i, idx in enumerate(remaining):
            x = arr[idx]
            # Check if x can be added: all sums x + y for y in current_subset must be new
            ok = True
            new_sums = set()
            for y in current_subset:
                s = x + y
                if s in current_sums:
                    ok = False
                    break
                new_sums.add(s)
            # Also check x + x
            if ok:
                s = x + x
                if s in current_sums or s in new_sums:
                    ok = False
                else:
                    new_sums.add(s)
            
            if ok:
                new_remaining = remaining[:i] + remaining[i+1:]
                backtrack(new_remaining, current_subset + [x], current_sums | new_sums)
            
            # Prune: if we've found a solution of size n, we can stop
            if best[0] == n:
                return
    
    backtrack(list(range(n)), [], set())
    return best[0]


def _candidate_sets():
    """Yield candidate 12-sets (tuples) to evaluate. Dense sets first."""
    seen = set()

    def emit(s):
        t = tuple(sorted(s))
        if len(t) == 12 and len(set(t)) == 12 and all(0 <= x <= 128 for x in t) and t not in seen:
            seen.add(t)
            yield t

    # 1) consecutive blocks starting at small offsets (densest)
    for start in range(0, 20):
        yield from emit(range(start, start + 12))
    
    # 2) low-value clusters with small gaps
    for s in [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
    ]:
        yield from emit(s)
    
    # 3) a few structured / spread sets (for safety)
    for s in [
        [0, 1, 4, 10, 12, 23, 25, 36, 38, 49, 51, 62],
        [0, 2, 3, 7, 12, 18, 25, 34, 44, 55, 67, 80],
        [0, 1, 3, 7, 12, 20, 30, 42, 56, 72, 90, 110],
    ]:
        yield from emit(s)


def construct():
    best_set = None
    best_size = 10**9
    
    for cand in _candidate_sets():
        if _time_left() <= 0:
            break
        s = _max_sidon_smart(cand)
        if s < best_size:
            best_size = s
            best_set = cand
        if best_size <= 4:
            break
    
    if best_set is None:
        return list(range(12))
    return list(best_set)
