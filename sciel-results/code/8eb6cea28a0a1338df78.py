import time
from collections import defaultdict

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin for 90s limit

    # Generate divisors
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # Precompute prefix sums for quick pruning
    prefix = [0] * (len(divisors) + 1)
    for i, d in enumerate(divisors):
        prefix[i + 1] = prefix[i] + d

    best_count = float('inf')
    best_set = None

    # Try increasing sizes k = 1, 2, ...
    # For each k, use DFS with strong pruning.
    for k in range(1, len(divisors) + 1):
        if time.time() > deadline:
            break

        # Early exit if k is larger than known best
        if k >= best_count:
            break

        def dfs(idx, count, current_sum, current_set):
            nonlocal best_count, best_set
            if time.time() > deadline:
                return False
            if count == k:
                if current_sum == target:
                    best_count = k
                    best_set = current_set.copy()
                    return True
                return False

            # Remaining items available
            remaining = len(divisors) - idx

            # Not enough items left to reach k
            if remaining < (k - count):
                return False

            # Pruning: even if we pick the largest possible remaining, can't reach target
            # Max possible additional sum = sum of top (k-count) remaining
            # But since sorted descending, the top (k-count) are divisors[idx : idx + (k-count)]
            # However, we might skip some. So max possible = sum of all remaining
            if current_sum + prefix[idx + remaining] - prefix[idx] < target - current_sum:
                # Actually, we want to know if we can reach target.
                # If even taking ALL remaining doesn't reach target, fail.
                if current_sum + (prefix[idx + remaining] - prefix[idx]) < target:
                    return False

            # Min possible additional sum = sum of smallest (k-count) remaining
            # Smallest are at the end.
            # But we are iterating from start.
            # Let's use a tighter bound if possible.
            # For now, stick with the "max sum" check.
            # Add a check for "too high" if we force pick?
            # No, we can skip.

            # Branch 1: Include divisors[idx]
            current_set.append(divisors[idx])
            if dfs(idx + 1, count + 1, current_sum + divisors[idx], current_set):
                current_set.pop()
                return True
            current_set.pop()

            # Branch 2: Exclude divisors[idx]
            # Check if enough items remain to fill k
            if (len(divisors) - (idx + 1)) >= (k - count):
                if dfs(idx + 1, count, current_sum, current_set):
                    return True

            return False

        # Start DFS for this k
        # Optimization: Start from index 0
        if dfs(0, 0, 0, []):
            break

    if best_set is None:
        # Fallback if no solution found (should not happen)
        # Try simple greedy or return empty
        best_set = []
        
    return best_set
