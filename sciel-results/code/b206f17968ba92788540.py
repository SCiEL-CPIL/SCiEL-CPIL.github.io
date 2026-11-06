"""
Program for large reciprocal-subsum-free sets.
Uses exact Fraction arithmetic and pruned DFS for subset sum checks.
Implements greedy construction with multiple random restarts and local search.
"""
import time
import math
import random
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a non-empty subset of items.
    items: list of Fractions
    target: Fraction
    Uses DFS with pruning based on suffix sums.
    """
    if not items:
        return target == 0
    
    # Sort items in descending order to prune more effectively? 
    # Actually, for subset sum, sorting descending allows us to see large values first.
    # But for pruning "current + suffix < target", we want to know the max possible sum.
    # Let's sort ascending and use prefix sums? Or just calculate total sum.
    
    total_sum = sum(items)
    if total_sum < target:
        return False
    if target < 0:
        return False
        
    # If target is 0, we need an empty subset. But the problem implies "subset of OTHER", 
    # and 1/x > 0, so target is never 0. 
    # However, if target is 0, empty set works. But we are checking 1/x, so target > 0.
    
    n = len(items)
    # Sort items. It doesn't matter for the existence of subset sum, but for pruning,
    # sorting descending might help if we check "can we reach target".
    # Let's use the standard DFS with memoization or just simple pruning.
    # Given n <= 23, simple DFS with pruning is likely fast enough.
    
    items_sorted = sorted(items, reverse=True)
    
    # Precompute suffix sums: suffix_sum[i] is the sum of items_sorted[i:]
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items_sorted[i]
        
    found = [False]
    
    def dfs(i, current_sum, is_non_empty):
        if found[0]:
            return
        if i == n:
            if current_sum == target and is_non_empty:
                found[0] = True
            return
        
        # Prune 1: Even if we take all remaining items, we can't reach target
        if current_sum + suffix_sums[i] < target:
            return
            
        # Prune 2: If current_sum already exceeds target (and items are positive)
        if current_sum > target:
            return

        # Option 1: Exclude items_sorted[i]
        dfs(i + 1, current_sum, is_non_empty)
        if found[0]:
            return
            
        # Option 2: Include items_sorted[i]
        new_sum = current_sum + items_sorted[i]
        if new_sum <= target:
            dfs(i + 1, new_sum, True)
            
    dfs(0, Fraction(0), False)
    return found[0]


def _is_subsum_free(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    selected: list of ints
    candidate: int
    """
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any non-empty subset of selected
    if selected:
        items = [Fraction(1, x) for x in selected]
        target = Fraction(1, candidate)
        if _can_form_sum(target, items):
            return False
    
    # Check for each element in selected: 1/x should not equal sum of reciprocals of any non-empty subset of (selected - {x}) + {candidate}
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        items = [Fraction(1, y) for y in others]
        target = Fraction(1, x)
        if _can_form_sum(target, items):
            return False
            
    return True


def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    trial_set = []
    for candidate in order:
        if _time_remaining() < 2.0:
            break
        if _is_subsum_free(trial_set, candidate):
            trial_set.append(candidate)
    return trial_set


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    """
    all_elements = list(range(1, 25))
    best_set = []
    
    # 1. Deterministic greedy: Large to Small
    if _time_remaining() > 5.0:
        s = _greedy_construct(sorted(all_elements, reverse=True))
        if len(s) > len(best_set):
            best_set = s

    # 2. Deterministic greedy: Small to Large
    if _time_remaining() > 5.0:
        s = _greedy_construct(sorted(all_elements))
        if len(s) > len(best_set):
            best_set = s
            
    # 3. Random restarts
    random.seed(42)
    # We can afford many restarts if the check is fast.
    # Let's try 100 restarts or until time runs out.
    
    for _ in range(200):
        if _time_remaining() < 10.0:
            break
            
        # Random order
        candidates = list(all_elements)
        random.shuffle(candidates)
        
        trial_set = _greedy_construct(candidates)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
            # Optional: If we find a significantly better set, we might want to try local search on it?
            # For now, just keep searching for larger sets.
            
    # 4. Local Search / Hill Climbing on the best set found
    # Try to swap elements to see if we can unlock adding more elements?
    # Actually, if we have size K, and we swap one out and one in, we still have size K.
    # To get size K+1, we need to find a set of size K+1.
    # Greedy with different orders is the main way to find different local optima.
    
    # Let's try a specific heuristic: 
    # If we are stuck at size K, try removing the smallest element and see if we can add two?
    # This is complex. Let's stick to more random restarts with different biases.
    
    # Biased random restarts:
    # 1. Mostly large numbers first
    # 2. Mostly small numbers first
    # 3. Alternating
    
    for _ in range(100):
        if _time_remaining() < 10.0:
            break
            
        # Strategy: Pick a random threshold, put large ones first, then small ones
        # Or just shuffle with a bias
        
        # Let's try sorting by random weights
        candidates = list(all_elements)
        # Assign random weights, but favor larger numbers slightly?
        # Or just pure random is usually sufficient for this size.
        
        # Let's try a "reverse greedy" approach:
        # Start with all elements, and remove ones that cause conflicts?
        # This is a different algorithm.
        
        # Let's just do more random shuffles.
        random.shuffle(candidates)
        trial_set = _greedy_construct(candidates)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
            
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
