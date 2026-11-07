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


class _CardinalityTracker:
    """Maintains sums_by_k incrementally for fast validity checks."""
    
    def __init__(self):
        self.sums_by_k = [set() for _ in range(61)]  # up to 60 elements
        self.sums_by_k[0].add(0)
        self.n = 0
    
    def add_element(self, x):
        """Add element x, return True if validity is preserved, False otherwise.
        If False, the state is not modified."""
        n = self.n
        # Check: for each k from 1 to n+1, the new sums at k are old_sums[k-1] + x
        # We need to ensure that no sum in (old_sums[k-1] + x) appears in old_sums[j] for j != k
        # AND that no sum in old_sums[k] (unchanged) appears in new_sums[j] for j != k
        # Actually, the full new sums_by_k[k] = old_sums[k] ∪ (old_sums[k-1] + x) for k>=1
        # And new_sums_by_k[0] = old_sums[0] = {0}
        
        # The conflict condition: there exists a sum s and two cardinalities k1 != k2
        # such that s is in new_sums_by_k[k1] and s is in new_sums_by_k[k2].
        
        # A sum s can be in new_sums_by_k[k] if:
        #   (a) s is in old_sums[k], OR
        #   (b) s - x is in old_sums[k-1] (i.e., s is in old_sums[k-1] + x)
        
        # So for each sum s, the set of cardinalities that can achieve it in the new state is:
        #   {k : s in old_sums[k]} ∪ {k : s - x in old_sums[k-1]}
        
        # We need this set to have size at most 1 for all s.
        
        # Efficient check: for each k from 0 to n, for each s in old_sums[k]:
        #   - s is already achievable at cardinality k (old)
        #   - s might also be achievable at cardinality k+1 in new state if s-x in old_sums[k]... 
        #     wait, new_sums_by_k[k+1] includes old_sums[k] + x, so s is in new_sums_by_k[k+1] iff s-x in old_sums[k]
        #   - s might be achievable at other cardinalities via the +x shift
        
        # Let me think differently. The new sums_by_k differs from old only in that:
        #   new_sums_by_k[k] = old_sums[k] ∪ (old_sums[k-1] + x) for 1 <= k <= n+1
        #   new_sums_by_k[0] = {0}
        
        # A conflict arises if there's a sum s and two different cardinalities k1, k2 where
        # s is in new_sums_by_k[k1] ∩ new_sums_by_k[k2].
        
        # Since old state was valid, conflicts can only involve at least one "new" sum, i.e.,
        # a sum in (old_sums[k-1] + x) for some k.
        
        # So: for each k from 1 to n+1, for each s in (old_sums[k-1] + x):
        #   Check if s is in old_sums[j] for some j != k. (If j == k, it's fine, it's just adding to existing.)
        #   Also check if s is in (old_sums[j-1] + x) for some j != k. (Two different +x shifts producing same sum)
        
        # But wait, s in (old_sums[k-1]+x) AND s in (old_sums[j-1]+x) means s-x in old_sums[k-1] AND s-x in old_sums[j-1],
        # which means s-x was achievable at two different cardinalities k-1 and j-1, contradicting old validity (unless k-1==j-1).
        # So the second type of conflict cannot happen if old was valid.
        
        # Therefore, we only need to check: for each k from 1 to n+1, for each s in (old_sums[k-1]+x),
        #   s is NOT in old_sums[j] for any j != k.
        
        # Equivalently: for each k from 1 to n+1, (old_sums[k-1] + x) ∩ (∪_{j != k} old_sums[j]) = ∅
        
        # Build the union of all old_sums except for each k... that's expensive.
        # Better: for each s in old_sums[k-1], check if s+x is in any old_sums[j] for j != k.
        
        # Precompute: for each sum, which cardinalities achieve it in old state.
        # Since old state is valid, each sum is in at most one old_sums[k].
        
        # So: for each k from 1 to n+1, for each s in old_sums[k-1]:
        #   if (s+x) is in old_sums[j] for some j != k, conflict.
        
        # To make this fast, build a dict: sum -> cardinality (or None if not present)
        sum_to_k = {}
        for k in range(n + 1):
            for s in self.sums_by_k[k]:
                # Old state is valid, so s should only be in one k
                if s in sum_to_k:
                    # This shouldn't happen if old state is valid
                    return False
                sum_to_k[s] = k
        
        for k in range(1, n + 2):
            if not self.sums_by_k[k - 1]:
                continue
            for s in self.sums_by_k[k - 1]:
                new_s = s + x
                if new_s in sum_to_k:
                    if sum_to_k[new_s] != k:
                        return False
                # If new_s is not in sum_to_k, it's a new sum at cardinality k, no conflict
                # (and it can't conflict with another new sum at different cardinality as argued above)
        
        # All checks passed, commit the addition
        for k in range(n + 1, 0, -1):
            if self.sums_by_k[k - 1]:
                for s in self.sums_by_k[k - 1]:
                    self.sums_by_k[k].add(s + x)
        self.n += 1
        return True
    
    def get_size(self):
        return self.n
    
    def reset(self):
        self.sums_by_k = [set() for _ in range(61)]
        self.sums_by_k[0].add(0)
        self.n = 0


def _greedy_build(candidates, tracker_class=_CardinalityTracker):
    """Greedy build using the given candidate ordering."""
    tracker = tracker_class()
    result = []
    for x in candidates:
        if tracker.n >= 60:
            break
        if tracker.add_element(x):
            result.append(x)
    return result, tracker


def _local_search(current, deadline):
    """
    Local search to maximize set size.
    1. Greedily add any valid element.
    2. If stuck, try swaps (remove u, add v) to escape local optima.
    """
    t_start = time.time()
    current_set = set(current)
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Try to add any element
        for x in range(1, 257):
            if x in current_set:
                continue
            if len(current) >= 60:
                break
            
            # Check if adding x is valid
            # We need to check if (S + x) is valid.
            # Using the tracker is efficient if we can "dry run" or rebuild.
            # Rebuilding is O(N * Sum). For N=60, Sum~15000, it's ~1M ops. Fast enough.
            
            candidate = current + [x]
            if _is_valid(candidate):
                current = candidate
                current_set.add(x)
                improved = True
                no_improve_count = 0
                break # Restart addition loop with new state
        
        if len(current) >= 60:
            break
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break # Give up on this local optimum
            
            # Phase 2: Try swaps
            # Randomly select a few swaps to try
            # To be fast, we don't try all 60*256 swaps every time, just a subset
            # Or try all if time permits
            
            # Generate list of u in current, v not in current
            # Shuffle and try a few
            swaps = [(u, v) for u in current for v in range(1, 257) if v not in current_set]
            random.shuffle(swaps)
            
            # Try up to 50 swaps
            for u, v in swaps[:50]:
                if time.time() > deadline - 2.0:
                    break
                
                # Remove u, add v
                candidate = [x for x in current if x != u] + [v]
                if _is_valid(candidate):
                    current = candidate
                    current_set.remove(u)
                    current_set.add(v)
                    improved = True
                    no_improve_count = 0
                    break # Restart loop to try additions again
    
    return current


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
    
    # Strategy 2: Random restarts
    # Increase attempts if time allows, but keep individual builds fast
    for attempt in range(500):
        if time_left() < 1.0:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        result, tracker = _greedy_build(candidates)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            # If we found a new best, we might want to run local search on it immediately?
            # No, let's collect bests and run local search on the top few at the end.
            # Or just run local search on the current best periodically.
    
    # Strategy 3: Local search on best set
    if time_left() > 5 and best_size > 0:
        current = best_set[:]
        improved = _local_search(current, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Also run local search on a few other good candidates
    if time_left() > 10:
        # Try local search on sets from different greedy orderings
        for ordering in orderings[:3]:
            if time_left() < 5:
                break
            result, _ = _greedy_build(ordering)
            if len(result) >= best_size - 1:  # Close to best
                improved = _local_search(result[:], deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]
    
    # Final validation
    if best_set and not _is_valid(best_set):
        best_set = [1]
    
    return list(best_set)
