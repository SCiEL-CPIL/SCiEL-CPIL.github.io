"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
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
    
    def count_subset_sums(elements):
        """Count number of subsets summing to each value up to target."""
        # dp[s] = number of subsets summing to s
        dp = [0] * (target + 1)
        dp[0] = 1
        for x in elements:
            for s in range(target - x, -1, -1):
                if dp[s] > 0:
                    dp[s + x] += dp[s]
        return dp
    
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
    
    # Start with full set and iteratively remove elements
    full_set = list(range(1, N + 1))
    
    def iterative_remove(removal_strategy, max_iter=500):
        """Iteratively remove elements to break all subset-sum-200 paths."""
        removed = set()
        remaining = list(range(1, N + 1))
        
        for _ in range(max_iter):
            if time.time() > deadline - 2:
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
            elif removal_strategy == 'most_frequent':
                # Count frequency of each element in all minimal subset sums
                # Approximate: count how many subsets of size <= len(subset) contain each element
                # For simplicity, just use the element that appears in the most "critical" positions
                # Use a heuristic: remove the element that, when removed, breaks the most subset sums
                # Count how many subset sums are "witnessed" by each element
                freq = {}
                for s in subset:
                    freq[s] = freq.get(s, 0) + 1
                to_remove = max(freq, key=freq.get)
            else:
                to_remove = max(subset)
            
            removed.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]
        
        return remaining
    
    # Try different removal strategies
    for strategy in ['largest', 'smallest', 'random', 'most_frequent']:
        if time.time() > deadline - 2:
            break
        cand = iterative_remove(strategy)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)
    
    if time.time() > deadline:
        return best_selected
    
    # Random restarts with random removal order
    random.seed(42)
    
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        cand = iterative_remove('random', max_iter=300)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)
    
    if time.time() > deadline:
        return best_selected
    
    # Local search: try to add back removed elements
    # Start from best_selected, try adding back elements one at a time
    all_elements = set(range(1, N + 1))
    current_set = set(best_selected)
    
    for _ in range(200):  # iterations of local search
        if time.time() > deadline - 1:
            break
        improved = False
        # Try adding back each element not in current_set
        for x in sorted(all_elements - current_set, key=lambda v: -v):
            if time.time() > deadline - 1:
                break
            trial_set = current_set | {x}
            # Check if adding x creates a subset sum of 200
            # We need to check if there's a subset of (current_set) that sums to 200-x
            # Combined with x, or a subset of current_set that sums to 200 (already false)
            # So we need: is there a subset of current_set that sums to 200 - x?
            if 200 - x >= 0:
                # Check if current_set has a subset summing to 200-x
                # Use DP
                elems = list(current_set)
                dp = [False] * (target + 1)
                dp[0] = True
                for e in elems:
                    for s in range(target - e, -1, -1):
                        if dp[s]:
                            dp[s + e] = True
                if not dp[200 - x]:
                    # Adding x is safe
                    current_set.add(x)
                    improved = True
                    break
        
        if not improved:
            break
    
    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
