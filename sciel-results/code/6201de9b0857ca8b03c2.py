"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def _get_divisors(n):
    """Return all positive divisors of n sorted in descending order."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort(reverse=True)
    return divs

def construct():
    target = 4979
    n = 5040
    divisors = _get_divisors(n)
    
    # The seed program uses a greedy approach from largest to smallest.
    # We want to minimize the number of divisors.
    # We can model this as a subset sum problem where we want the smallest subset.
    # Since n=5040 is small and target=4979 is close to n, the number of divisors
    # in the solution should be small (likely < 10).
    
    # Strategy:
    # 1. Try to find a solution with k=1, k=2, ... up to some limit (e.g., 10 or 12).
    # 2. For each k, use a recursive backtracking search with pruning.
    #    Pruning:
    #    - If current sum > target, prune.
    #    - If current sum + sum of remaining largest possible divisors < target, prune.
    #    - Since we are iterating in descending order, we can also prune if the
    #      minimum possible sum (using the smallest available divisors for remaining slots) > target?
    #      Actually, simpler: just standard subset sum backtracking.
    #    
    #    To speed up, we can pre-compute suffix sums of the sorted divisors.
    #    suffix_sum[i] = sum(divisors[i:])
    #    If at index i, current_sum + suffix_sum[i] < target, we can't reach target.
    #    If current_sum > target, prune.
    #    
    #    We are picking exactly k items.
    #    So at index i, if we have picked p items, we need k-p items from divisors[i:].
    #    Min possible sum: sum of smallest (k-p) divisors from i onwards.
    #    Max possible sum: sum of largest (k-p) divisors from i onwards.
    #    If target - current_sum > MaxPossible, prune.
    #    If target - current_sum < MinPossible, prune.
    
    num_divs = len(divisors)
    
    # Precompute suffix sums for max pruning
    suffix_sum = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # Precompute prefix sums of smallest? 
    # Actually, for "min possible sum", we need the sum of the smallest (k-p) elements in divisors[i:].
    # This is a bit harder to precompute efficiently for all i and k-p.
    # However, since k is small, we can just check if target - current_sum >= 0.
    # The main pruning is:
    # 1. current_sum <= target
    # 2. current_sum + suffix_sum[i] >= target (enough total value remaining)
    # 3. We need exactly k-p items. The maximum value we can add is suffix_sum[i].
    #    The minimum value we can add is 0 (if we skip all, but we must pick k-p).
    #    Actually, the minimum value we *must* add if we pick k-p items is the sum of the
    #    smallest k-p items in divisors[i:].
    #    Let's just use the standard pruning:
    #    - current_sum <= target
    #    - current_sum + suffix_sum[i] >= target
    #    - i + (k - p) <= num_divs (enough items left to pick)
    
    start_time = time.time()
    deadline = start_time + 45.0  # 15s safety margin
    
    best_solution = None
    
    # Try k from 1 to 12
    for k in range(1, 13):
        if time.time() > deadline:
            break
            
        found = False
        current_set = []
        
        def backtrack(idx, current_sum, picked):
            nonlocal found, best_solution
            if time.time() > deadline:
                return False
                
            if picked == k:
                if current_sum == target:
                    found = True
                    best_solution = current_set[:]
                    return True
                return False
                
            # If we run out of divisors
            if idx >= num_divs:
                return False
                
            # Not enough items left to pick k - picked
            if num_divs - idx < k - picked:
                return False
                
            # Pruning:
            # Max possible sum we can achieve is current_sum + suffix_sum[idx]
            if current_sum + suffix_sum[idx] < target:
                return False
                
            # Min possible sum? 
            # We must pick k-picked items from idx onwards.
            # The smallest sum is sum of the smallest (k-picked) items in divisors[idx:].
            # This is expensive to compute on the fly. 
            # However, since divisors are sorted descending, the smallest are at the end.
            # We can approximate or just rely on the max pruning and the fact that k is small.
            # Another pruning: if current_sum > target, return False (already handled by max sum check? No, max sum check allows current_sum > target if suffix is negative? No, suffix is positive. So if current_sum > target, current_sum + suffix_sum[idx] > target, so it passes. We need explicit check.)
            if current_sum > target:
                return False
                
            # Option 1: Pick divisors[idx]
            if divisors[idx] <= target - current_sum:
                current_set.append(divisors[idx])
                if backtrack(idx + 1, current_sum + divisors[idx], picked + 1):
                    return True
                current_set.pop()
            
            # Option 2: Skip divisors[idx]
            # Check if it's possible to reach target without this divisor
            # We still need k-picked items from idx+1 onwards
            if num_divs - (idx + 1) >= k - picked:
                if backtrack(idx + 1, current_sum, picked):
                    return True
                    
            return False
        
        # Optimization: Start from the largest divisors.
        # We can also limit the search space.
        # If k is small, we don't need to look at the smallest divisors early on?
        # The backtracking handles the order.
        
        if backtrack(0, 0, 0):
            if best_solution:
                return best_solution

    # Fallback to greedy if no small k found (should not happen for this instance)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    if remainder == 0:
        return chosen
    else:
        return []

# EVOLVE-BLOCK-END
