"""Rewritten program with stronger search for large sets without subset averages."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s limit
    n = 512
    target_size = 48
    
    def _subset_sum_bitset(arr, max_sum):
        """Return list of bitsets where bitsets[k] has bit s set if some subset of arr of size k sums to s."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        mask = (1 << (max_sum + 1)) - 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val) & mask
        return dp
    
    def _can_form_sum(arr, target_sum, min_size, max_size):
        """Check if any subset of arr with size in [min_size, max_size] sums to target_sum."""
        if target_sum < 0:
            return False
        m = len(arr)
        if min_size > m:
            return False
        max_size = min(max_size, m)
        if max_size < min_size:
            return False
        # Quick sum check
        if target_sum > sum(arr):
            return False
        if target_sum < min_size * min(arr) if arr else 0:
            return False
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        for k in range(min_size, max_size + 1):
            if (dp[k] >> target_sum) & 1:
                return True
        return False
    
    def _is_valid_incremental(existing, x):
        """Check if adding x to existing set keeps it valid."""
        if not existing:
            return True
        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, len(existing) + 1):
            target = k * x
            if target > sum(existing):
                break
            if _can_form_sum(existing, target, k, k):
                return False
        # Check 2: no element a in existing is the mean of a subset of existing∪{x} of size >= 2 that includes x
        for a in existing:
            for t_size in range(1, len(existing) + 1):
                target = (t_size + 1) * a - x
                if target < 0:
                    continue
                if target > sum(existing):
                    break
                if _can_form_sum(existing, target, t_size, t_size):
                    return False
        return True
    
    def _greedy_from_order(order, selected_start=None):
        """Try to build a valid set by greedily adding elements in the given order."""
        if selected_start is None:
            selected = []
        else:
            selected = list(selected_start)
        for x in order:
            if x in selected:
                continue
            if _is_valid_incremental(selected, x):
                selected.append(x)
                if len(selected) >= target_size:
                    break
        return selected
    
    def _local_search(selected, max_iterations=50):
        """Try to improve the set by swapping elements."""
        selected = list(selected)
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try removing one element and adding a better one
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                candidate_remove = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                # Try to add elements from 1..512 not in remaining
                for x in range(1, n + 1):
                    if x in remaining or x == candidate_remove:
                        continue
                    if _is_valid_incremental(remaining, x):
                        new_selected = remaining + [x]
                        if len(new_selected) >= len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return selected
    
    # Strategy 1: Greedy from natural order
    best = []
    order = list(range(1, n + 1))
    best = _greedy_from_order(order)
    
    if len(best) >= target_size:
        return best
    
    # Strategy 2: Try different random orderings
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 3: Start with a known good core and expand
    # Try starting with large numbers (they're less likely to be averages)
    order_desc = list(range(n, 0, -1))
    result = _greedy_from_order(order_desc)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 4: Local search on best found
    if len(best) < target_size:
        improved = _local_search(best, max_iterations=20)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 5: Try biased random orders (favoring numbers with certain properties)
    for trial in range(30):
        if time.time() > deadline - 5.0:
            break
        random.seed(1000 + trial)
        # Bias towards numbers that are primes or have few divisors
        weights = []
        for x in range(1, n + 1):
            # Count divisors
            div_count = 0
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    div_count += 1
                    if d != x // d:
                        div_count += 1
            # Fewer divisors = higher weight
            weights.append(1.0 / max(div_count, 1))
        
        order = list(range(1, n + 1))
        # Sample without replacement using weights
        selected_order = []
        remaining = list(range(n))
        remaining_weights = weights[:]
        for _ in range(n):
            total_w = sum(remaining_weights)
            r = random.random() * total_w
            cum = 0
            for idx, w in enumerate(remaining_weights):
                cum += w
                if cum >= r:
                    selected_order.append(remaining[idx])
                    remaining.pop(idx)
                    remaining_weights.pop(idx)
                    break
        result = _greedy_from_order(selected_order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    return best
# EVOLVE-BLOCK-END
