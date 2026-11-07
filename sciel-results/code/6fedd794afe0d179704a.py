"""Cardinality-separated subset sums: fast incremental DP + local search.

Key improvements over the previous version:
- Much faster incremental validity check using interval tracking per cardinality.
- More aggressive local search: multiple restarts, swap-based improvements.
- Structured construction that aims for wide gaps between cardinality sum ranges.
- Tighter deadline management to use all available time productively.
"""
import time
import random

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill


def _subset_sums_by_cardinality(S):
    """Return list: sums_by_k[k] = set of achievable sums with exactly k elements."""
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k


def _is_valid(S):
    """Check no two subsets of different cardinalities share a sum."""
    if not S:
        return True
    sums_by_k = _subset_sums_by_cardinality(S)
    # For each sum, record which cardinalities achieve it
    sum_to_ks = {}
    for k, sums in enumerate(sums_by_k):
        for s in sums:
            if s in sum_to_ks:
                sum_to_ks[s].add(k)
            else:
                sum_to_ks[s] = {k}
    for s, ks in sum_to_ks.items():
        if len(ks) > 1:
            return False
    return True


class _BitsetTracker:
    """Maintains sums_by_k as bitsets for O(1) collision checks."""
    
    def __init__(self, max_sum=15360):
        # max_sum = 60 * 256 = 15360
        self.max_sum = max_sum
        # Each cardinality k has a bitset where bit i is set if sum i is achievable
        self.sums_by_k = [0] * 61  # up to 60 elements
        self.sums_by_k[0] = 1  # sum 0 achievable at cardinality 0
        self.n = 0
        # Mask for valid sums (0 to max_sum)
        self.mask = (1 << (max_sum + 1)) - 1
    
    def add_element(self, x):
        """Add element x, return True if validity is preserved, False otherwise.
        If False, the state is not modified."""
        n = self.n
        
        # Build the union of all old sums across all cardinalities
        # Since old state is valid, each sum is in at most one cardinality
        all_sums = 0
        for k in range(n + 1):
            all_sums |= self.sums_by_k[k]
        
        # For each k from 1 to n+1, new sums at k include (old_sums[k-1] << x)
        # We need to check that (old_sums[k-1] << x) doesn't intersect with
        # any old_sums[j] for j != k
        
        # Precompute: for each sum s, which cardinality achieves it
        # Since old state is valid, we can build a mapping
        # But with bitsets, we check collisions differently:
        
        # For a candidate element x:
        # new_sums_by_k[k] = old_sums_by_k[k] | (old_sums_by_k[k-1] << x)
        # Collision if there exist k1 != k2 such that:
        #   (new_sums_by_k[k1] & new_sums_by_k[k2]) != 0
        # Since old state was valid, old_sums_by_k[k1] & old_sums_by_k[k2] = 0 for k1 != k2
        # So collision only involves new terms:
        #   (old_sums_by_k[k1-1] << x) & (old_sums_by_k[k2]) != 0 for k1 != k2
        #   or (old_sums_by_k[k1-1] << x) & (old_sums_by_k[k2-1] << x) != 0
        # The second case: (A << x) & (B << x) = (A & B) << x, which is 0 if A & B = 0
        # So we only need to check: for all k1 != k2, (old_sums_by_k[k1-1] << x) & old_sums_by_k[k2] == 0
        
        # More efficiently: for each k from 1 to n+1:
        #   shifted_k = (self.sums_by_k[k-1] << x) & self.mask
        #   Check shifted_k & (all_sums ^ self.sums_by_k[k]) == 0
        #   i.e., shifted_k should only have bits in positions that are either
        #   already in self.sums_by_k[k] or not in all_sums at all
        
        for k in range(1, n + 2):
            if self.sums_by_k[k - 1] == 0:
                continue
            shifted = (self.sums_by_k[k - 1] << x) & self.mask
            # Collision if shifted has any bit set that's in all_sums but not in self.sums_by_k[k]
            if shifted & (all_sums ^ self.sums_by_k[k]):
                return False
        
        # All checks passed, commit the addition
        for k in range(n + 1, 0, -1):
            if self.sums_by_k[k - 1]:
                self.sums_by_k[k] |= (self.sums_by_k[k - 1] << x) & self.mask
        self.n += 1
        return True
    
    def get_size(self):
        return self.n
    
    def reset(self):
        self.sums_by_k = [0] * 61
        self.sums_by_k[0] = 1
        self.n = 0


def _greedy_build(candidates, tracker_class=_BitsetTracker):
    """Greedy build using the given candidate ordering."""
    tracker = tracker_class()
    result = []
    for x in candidates:
        if tracker.n >= 60:
            break
        if tracker.add_element(x):
            result.append(x)
    return result, tracker


def _local_search(current, tracker, max_time=5.0):
    """Try to improve the current set via additions and swaps."""
    t_start = time.time()
    current_set = set(current)
    
    # Try adding any element not in current
    for x in range(1, 257):
        if time.time() - t_start > max_time:
            break
        if x in current_set:
            continue
        if tracker.n >= 60:
            break
        if tracker.add_element(x):
            current.append(x)
            current_set.add(x)
        else:
            # Need to revert - but our tracker doesn't support undo easily
            # Let's use a different approach: build a new tracker
            pass
    
    # Actually, the tracker doesn't support removal. Let me restructure.
    # For local search, we'll rebuild the tracker each time we want to try a modification.
    # This is slower but correct.
    
    # Alternative: just do multiple greedy passes with different orderings
    return current


def _try_improve_with_removal(current, t_start, deadline):
    """Try removing one element and then greedily adding others."""
    best = current[:]
    best_size = len(current)
    
    for i in range(len(current)):
        if time.time() > deadline - 1.0:
            break
        # Try removing element i
        candidate = current[:i] + current[i+1:]
        tracker = _BitsetTracker()
        for x in candidate:
            if not tracker.add_element(x):
                break  # Shouldn't happen if current was valid
        else:
            # Successfully rebuilt, now try adding
            current_set = set(candidate)
            for x in range(1, 257):
                if time.time() > deadline - 0.5:
                    break
                if x in current_set:
                    continue
                if tracker.n >= 60:
                    break
                if tracker.add_element(x):
                    candidate.append(x)
                    current_set.add(x)
            
            if len(candidate) > best_size:
                best_size = len(candidate)
                best = candidate[:]
    
    return best


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    def time_left():
        return deadline - time.time()
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Multiple greedy passes with different orderings
    # The key insight: we want elements that create wide gaps between cardinality sum ranges.
    # Small elements first, then large ones, tends to work well.
    
    orderings = [
        list(range(1, 257)),                          # Ascending
        list(range(256, 0, -1)),                       # Descending
        list(range(1, 20)) + list(range(100, 257)),   # Small then large
        [1, 2, 4, 8, 16, 32, 64, 128] + list(range(3, 257)),  # Powers of 2 first
        list(range(1, 50)) + list(range(50, 257, 2)), # Odd spacing
    ]
    
    for ordering in orderings:
        if time_left() < 2.0:
            break
        result, tracker = _greedy_build(ordering)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 2: Random restarts - with bitsets, each check is O(1) per cardinality
    # so we can do many more restarts
    for attempt in range(5000):
        if time_left() < 1.0:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        result, tracker = _greedy_build(candidates)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 3: Local search on best set
    if time_left() > 5 and best_size > 0:
        current = best_set[:]
        # Try to add elements
        tracker = _BitsetTracker()
        for x in current:
            tracker.add_element(x)
        
        for x in range(1, 257):
            if time_left() < 1.0:
                break
            if x in current:
                continue
            if tracker.n >= 60:
                break
            if tracker.add_element(x):
                current.append(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Try removal-based improvement
        if time_left() > 10:
            improved = _try_improve_with_removal(best_set, t_start, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Final validation
    if best_set and not _is_valid(best_set):
        best_set = [1]
    
    return list(best_set)
