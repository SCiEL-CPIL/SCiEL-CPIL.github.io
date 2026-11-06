"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
def _precompute_conflicts():
    """
    For each element a in [1..24], find all subsets T of [1..24] \\ {a}
    such that 1/a = sum(1/t for t in T).
    Return dict: a -> list of frozensets T.
    """
    n = 24
    universe = list(range(1, n + 1))
    conflicts = {a: [] for a in universe}
    
    # For each target a, we need subsets T of universe \ {a} where sum of reciprocals = 1/a
    # Since 1/a >= 1/24, and we're summing reciprocals of distinct elements from [1..24]\{a},
    # the maximum sum of any subset is sum(1/i for i in [1..24] if i != a)
    # Minimum sum of any non-empty subset is 1/24 (or 1/23 if a=24, etc.)
    
    # We can enumerate subsets by size. The sum of reciprocals for subset T of size k
    # is at least 1/24 + 1/23 + ... (k smallest reciprocals from available)
    # and at most 1/1 + 1/2 + ... (k largest reciprocals from available)
    
    # For a given a, target = 1/a.
    # A subset T must satisfy sum(1/t for t in T) = 1/a.
    # Since all elements are positive, T must be non-empty (reciprocal-subsum implies at least one other element).
    # Actually, the problem says "subset of the OTHER selected elements", so T is non-empty subset of S\{a}.
    
    # Let's enumerate all non-empty proper subsets of [1..24]\{a} and check.
    # But that's 2^23 - 1 subsets per a, which is 8M per a, 24*8M = 200M total. Too slow in Python.
    
    # Better: enumerate by size. For a given a and target size k, we can use DFS with pruning.
    # Target sum = 1/a.
    # The elements available are [1..24]\{a}.
    # Sort available elements. For a subset of size k, the minimum possible sum is the sum of the k smallest available reciprocals,
    # and the maximum is the sum of the k largest available reciprocals.
    
    # Actually, let's think about this differently. The equation is:
    # 1/a = 1/t1 + 1/t2 + ... + 1/tk where t1, ..., tk are distinct elements from [1..24]\{a}.
    # Multiply both sides by a: 1 = a/t1 + a/t2 + ... + a/tk
    # So we need a/ti to be rational numbers that sum to 1.
    
    # Let's just do a DFS for each a, enumerating subsets in increasing size order, with pruning.
    
    def dfs_for_a(a, available, target, current_sum, current_subset, min_idx, max_size):
        """DFS to find subsets of 'available' (indexed from min_idx) that sum to target."""
        if current_sum == target and len(current_subset) > 0:
            conflicts[a].append(frozenset(current_subset))
            return
        if current_sum > target:
            return
        if len(current_subset) >= max_size:
            return
        if min_idx >= len(available):
            return
        
        # Pruning: even if we take all remaining elements, can we reach target?
        # Max possible additional sum = sum of 1/x for x in available[min_idx:]
        # But we also need exactly the remaining amount, and we can choose any subset.
        # Let's compute the max sum achievable from available[min_idx:] with at most (max_size - len(current_subset)) elements.
        # The max sum is achieved by taking the largest elements (smallest reciprocals? No, largest reciprocals = smallest elements).
        # Wait, 1/x is decreasing in x. So the largest reciprocal is 1/1, then 1/2, etc.
        # So to maximize sum, we take the smallest x values.
        
        remaining_needed = target - current_sum
        if remaining_needed <= 0:
            if remaining_needed == 0:
                pass  # already handled above
            return
        
        # Can we reach remaining_needed with the available elements from min_idx onwards?
        # The maximum sum we can get is the sum of the smallest (max_size - len(current_subset)) elements in available[min_idx:].
        # But we need to be careful: we want to check if it's possible.
        
        # Let's just recurse and rely on the current_sum > target check for pruning.
        # Add a stronger prune: if current_sum + sum of all remaining available reciprocals < target, prune.
        
        # Compute max possible sum from available[min_idx:]
        # Actually, let's compute the sum of all remaining reciprocals. If current_sum + that sum < target, prune.
        remaining_sum = sum(Fraction(1, x) for x in available[min_idx:])
        if current_sum + remaining_sum < target:
            return
        
        # Option 1: don't take available[min_idx]
        dfs_for_a(a, available, target, current_sum, current_subset, min_idx + 1, max_size)
        
        # Option 2: take available[min_idx]
        x = available[min_idx]
        new_sum = current_sum + Fraction(1, x)
        if new_sum <= target:
            dfs_for_a(a, available, target, new_sum, current_subset + [x], min_idx + 1, max_size)
    
    for a in universe:
        available = [x for x in universe if x != a]
        available.sort()
        target = Fraction(1, a)
        
        # What's the maximum size of a subset T? 
        # The minimum reciprocal is 1/24. So k * 1/24 <= 1/a => k <= 24/a.
        # For a=1, k <= 24. But available has 23 elements.
        # For a=2, k <= 12.
        # etc.
        max_size = min(len(available), int(24 / a) if a > 0 else 24)
        # But we need at least 1 element.
        max_size = max(1, max_size)
        
        # Also, the minimum sum of k elements is the sum of the k smallest reciprocals (largest x values).
        # But for now, let's just use the DFS with the above pruning.
        
        # Start DFS
        dfs_for_a(a, available, target, Fraction(0), [], 0, max_size)
    
    return conflicts


def _validate_set(S, conflicts):
    """Check if set S is reciprocal-subsum-free."""
    S = set(S)
    for a in S:
        others = S - {a}
        for T in conflicts[a]:
            if T.issubset(others):
                return False
    return True


def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin
    
    n = 24
    universe = list(range(1, n + 1))
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(range(13, 25))
    
    # Greedy heuristic: start with all elements, remove the one that causes the most conflicts
    # Or: build up greedily, adding elements that don't create conflicts.
    
    # Strategy 1: Try to find a large valid set using greedy + local search.
    # Start with a random order or sorted order, add elements if they don't conflict.
    
    # Let's try: sort by "conflict potential" - elements with fewer conflicts first?
    # Or just try all elements and see.
    
    # Actually, let's use a different approach: start with the full set and remove elements to resolve conflicts.
    # Or use a simple greedy: process elements in some order, add if valid.
    
    # Let's try multiple greedy strategies and pick the best.
    
    def greedy_by_order(order):
        S = set()
        for a in order:
            S_candidate = S | {a}
            valid = True
            for b in S_candidate:
                others = S_candidate - {b}
                for T in conflicts[b]:
                    if T.issubset(others):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S = S_candidate
        return list(S)
    
    # Try different orders
    best = []
    
    # Natural order
    result = greedy_by_order(universe)
    if len(result) > len(best):
        best = result
    
    # Reverse order
    result = greedy_by_order(list(reversed(universe)))
    if len(result) > len(best):
        best = result
    
    # Order by number of conflicts (fewest first)
    conflict_counts = {a: len(conflicts[a]) for a in universe}
    order = sorted(universe, key=lambda x: conflict_counts[x])
    result = greedy_by_order(order)
    if len(result) > len(best):
        best = result
    
    # Order by number of conflicts (most first)
    order = sorted(universe, key=lambda x: -conflict_counts[x])
    result = greedy_by_order(order)
    if len(result) > len(best):
        best = result
    
    # Order by element value (largest first)
    order = list(reversed(universe))
    result = greedy_by_order(order)
    if len(result) > len(best):
        best = result
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best if best else list(range(13, 25))
    
    # Try to improve: for each element not in best, try adding it and removing conflicting elements.
    # Or try removing one element from best and see if we can add two.
    
    # Let's do a simple local search: try swapping one element out for another.
    best_set = set(best)
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        current_size = len(best_set)
        for a in universe:
            if a in best_set:
                continue
            # Try adding a, then removing elements to fix conflicts
            S_candidate = best_set | {a}
            # Find all conflicts in S_candidate
            conflicting = set()
            for b in S_candidate:
                others = S_candidate - {b}
                for T in conflicts[b]:
                    if T.issubset(others):
                        conflicting.add(b)
                        for t in T:
                            conflicting.add(t)
            # Remove conflicting elements, but keep a
            S_new = S_candidate - conflicting
            if a not in S_new:
                S_new.add(a)
            # Check if valid
            if _validate_set(S_new, conflicts) and len(S_new) > current_size:
                best_set = S_new
                best = list(S_new)
                improved = True
        
        # Also try removing one element and adding two
        for a in list(best_set):
            S_without = best_set - {a}
            for b in universe:
                if b in S_without:
                    continue
                S_candidate = S_without | {b}
                if _validate_set(S_candidate, conflicts) and len(S_candidate) > current_size:
                    best_set = S_candidate
                    best = list(S_candidate)
                    improved = True
                for c in universe:
                    if c in S_candidate or c == b:
                        continue
                    S_candidate2 = S_without | {b, c}
                    if _validate_set(S_candidate2, conflicts) and len(S_candidate2) > current_size:
                        best_set = S_candidate2
                        best = list(S_candidate2)
                        improved = True
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best if best else list(range(13, 25))
    
    # Try a more systematic search: use the precomputed conflicts to do a branch and bound.
    # But let's first see if we can find a larger set by trying all subsets of size 13, 14, etc.
    # Actually, the initial answer is 12 elements (13-24). Let's try to find 13 or more.
    
    # Let's try all subsets of size 13 from a reduced candidate set.
    # But C(24,13) is too large.
    
    # Let's just return what we have.
    return best if best else list(range(13, 25))
# EVOLVE-BLOCK-END
