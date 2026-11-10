"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    N = 100
    target = 200
    
    def greedy_largest_first():
        selected = []
        reachable = {0}
        for x in range(N, 0, -1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable
    
    def greedy_smallest_first():
        selected = []
        reachable = {0}
        for x in range(1, N + 1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable
    
    def find_subset_sum_fast(elements):
        """Find a subset of elements summing to target, return it or None.
        Uses array-based DP for speed."""
        n = len(elements)
        # dp[s] = index of element that last contributed to reach sum s, or -1
        dp = [-1] * (target + 1)
        dp[0] = -1  # base case: sum 0 is reachable with no elements
        parent = {}  # (sum, elem_idx) -> prev_sum
        
        for i, x in enumerate(elements):
            # Iterate backwards to avoid using the same element twice
            for s in range(target - x, -1, -1):
                if dp[s] != -1 and dp[s + x] == -1:
                    dp[s + x] = i
                    parent[(s + x, i)] = s
            if dp[target] != -1:
                # Reconstruct
                subset = []
                cur = target
                while cur > 0:
                    i = dp[cur]
                    subset.append(elements[i])
                    cur = parent[(cur, i)]
                return subset
        return None
    
    def compute_reachable_sums(elements):
        """Compute set of all subset sums up to target using numpy for speed."""
        reachable = np.zeros(target + 1, dtype=bool)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            # Shift reachable sums by x and OR
            reachable[x:] |= reachable[:-x]
        return reachable
    
    def iterative_remove_optimal(removal_strategy='critical', max_iter=500):
        """Iteratively remove elements to break all subset-sum-200 paths.
        Uses critical element removal: remove the element that appears in the most
        subset-sum-200 solutions."""
        remaining = list(range(1, N + 1))
        removed = set()
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            subset = find_subset_sum_fast(remaining)
            if subset is None:
                break
            
            if removal_strategy == 'largest':
                to_remove = max(subset)
            elif removal_strategy == 'smallest':
                to_remove = min(subset)
            elif removal_strategy == 'random':
                to_remove = random.choice(subset)
            elif removal_strategy == 'critical':
                # Find which element in the subset is most "critical"
                # by checking how many subset sums would be broken if we remove it
                # Approximation: remove the median element of the subset
                sorted_subset = sorted(subset)
                mid = len(sorted_subset) // 2
                to_remove = sorted_subset[mid]
            else:
                to_remove = max(subset)
            
            removed.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]
        
        return remaining
    
    # Greedy from large to small:
    selected, reachable = greedy_largest_first()
    best_selected = list(selected)
    best_size = len(best_selected)
    
    if time.time() > deadline:
        return best_selected
    
    # Try greedy from small to large:
    sel2, reach2 = greedy_smallest_first()
    if len(sel2) > best_size:
        best_selected = list(sel2)
        best_size = len(best_selected)
    
    if time.time() > deadline:
        return best_selected
    
    # Try different removal strategies
    for strategy in ['largest', 'smallest', 'critical']:
        if time.time() > deadline - 3:
            break
        cand = iterative_remove_optimal(strategy, max_iter=300)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)
    
    if time.time() > deadline:
        return best_selected
    
    # Random restarts with random removal order
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline - 3:
            break
        cand = iterative_remove_optimal('random', max_iter=200)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)
    
    if time.time() > deadline:
        return best_selected
    
    # Local search: try to add back removed elements
    # Start from best_selected, try adding back elements one at a time
    all_elements = set(range(1, N + 1))
    current_set = set(best_selected)
    
    # Precompute reachable sums for current_set using numpy
    def can_add_element(x, current_reachable):
        """Check if adding x to current_set keeps it free of subset sum 200.
        We need to check if there's a subset of current_set that sums to 200-x."""
        if x > target:
            return False
        # Check if current_reachable has sum 200-x
        if 200 - x <= target and current_reachable[200 - x]:
            return False
        return True
    
    # Compute reachable sums for current_set
    current_reachable = compute_reachable_sums(list(current_set))
    
    for _ in range(100):  # iterations of local search
        if time.time() > deadline - 2:
            break
        improved = False
        # Try adding back each element not in current_set, in random order
        candidates = list(all_elements - current_set)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if can_add_element(x, current_reachable):
                # Adding x is safe
                current_set.add(x)
                # Update reachable sums
                current_reachable[x:] |= current_reachable[:-x]
                improved = True
                break
        
        if not improved:
            break
    
    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)
    
    # Second pass local search with different ordering
    current_set = set(best_selected)
    current_reachable = compute_reachable_sums(list(current_set))
    
    for _ in range(100):  # iterations of local search
        if time.time() > deadline - 1:
            break
        improved = False
        # Try adding back each element not in current_set, in descending order
        candidates = sorted(all_elements - current_set, reverse=True)
        
        for x in candidates:
            if time.time() > deadline - 1:
                break
            if can_add_element(x, current_reachable):
                # Adding x is safe
                current_set.add(x)
                # Update reachable sums
                current_reachable[x:] |= current_reachable[:-x]
                improved = True
                break
        
        if not improved:
            break
    
    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
