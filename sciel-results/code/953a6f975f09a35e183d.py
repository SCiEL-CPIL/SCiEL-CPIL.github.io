import time
import random

def _has_subset_sum_200(elements):
    """Check if any subset of elements sums to 200."""
    reachable = {0}
    for x in elements:
        if x > 200:
            continue
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 in new_sums:
            return True
        reachable |= new_sums
    return 200 in reachable

def _find_subset_sum_200(elements):
    """Find a subset of elements that sums to 200, or return None."""
    # DP with backtracking
    n = len(elements)
    # dp[i][s] = can we achieve sum s using subset of elements[:i]
    # But we want to reconstruct, so use a different approach
    
    # Use bitset with parent tracking
    # Simpler: DFS with memoization
    from functools import lru_cache
    
    elements_sorted = sorted(elements, reverse=True)
    n = len(elements_sorted)
    
    @lru_cache(maxsize=None)
    def dfs(idx, target):
        if target == 0:
            return []
        if idx >= n or target < 0:
            return None
        # Try including elements_sorted[idx]
        if elements_sorted[idx] <= target:
            result = dfs(idx + 1, target - elements_sorted[idx])
            if result is not None:
                return [elements_sorted[idx]] + result
        # Try excluding
        return dfs(idx + 1, target)
    
    return dfs(0, 200)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_size = 0
    best_set = None
    
    # Initial: try greedy removal
    all_elements = list(range(1, 101))
    
    # Start with all elements, iteratively remove
    removed = set()
    current = set(all_elements)
    
    # Phase 1: Greedily remove elements to make valid
    while _has_subset_sum_200(current):
        if time.time() > deadline - 10:
            break
        subset = _find_subset_sum_200(list(current))
        if subset is None:
            break
        # Try removing each element in the subset, pick the one that minimizes further damage
        # For simplicity, remove the largest element in the subset (heuristic)
        # Or: try all, pick the one that leads to valid state with fewest total removals
        best_choice = None
        best_remaining = None
        for elem in subset:
            test_set = current - {elem}
            # Count how many more removals needed (approximate by checking validity)
            # Just pick the largest for now
            pass
        # Heuristic: remove the largest element in the subset
        elem_to_remove = max(subset)
        current.remove(elem_to_remove)
        removed.add(elem_to_remove)
    
    if _has_subset_sum_200(current) == False:
        best_size = len(current)
        best_set = set(current)
    
    # Phase 2: Try to add back removed elements
    if best_set is not None:
        for elem in sorted(removed, reverse=True):
            if time.time() > deadline - 5:
                break
            test_set = best_set | {elem}
            if not _has_subset_sum_200(test_set):
                best_set = test_set
                removed.discard(elem)
    
    # Phase 3: Local search - try swapping
    if best_set is not None:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for elem_in in sorted(best_set):
                if time.time() > deadline - 5:
                    break
                for elem_out in sorted(removed):
                    if time.time() > deadline - 3:
                        break
                    test_set = (best_set - {elem_in}) | {elem_out}
                    if not _has_subset_sum_200(test_set):
                        best_set = test_set
                        removed.discard(elem_out)
                        removed.add(elem_in)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Phase 4: Random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(restart)
        current = set(all_elements)
        removed = set()
        
        # Randomized greedy removal
        while _has_subset_sum_200(current):
            if time.time() > deadline - 10:
                break
            subset = _find_subset_sum_200(list(current))
            if subset is None:
                break
            elem_to_remove = random.choice(subset)
            current.remove(elem_to_remove)
            removed.add(elem_to_remove)
        
        if not _has_subset_sum_200(current):
            # Try to add back
            for elem in sorted(removed, reverse=True):
                if time.time() > deadline - 5:
                    break
                test_set = current | {elem}
                if not _has_subset_sum_200(test_set):
                    current = test_set
                    removed.discard(elem)
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
    
    if best_set is None:
        # Fallback: greedy from seed
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    return list(best_set)
