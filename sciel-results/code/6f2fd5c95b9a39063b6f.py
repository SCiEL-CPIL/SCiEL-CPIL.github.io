import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 80.0

    # Generate divisors
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # Precompute prefix sums for pruning
    prefix = [0] * (len(divisors) + 1)
    for i, d in enumerate(divisors):
        prefix[i + 1] = prefix[i] + d

    best_set = None

    # Try increasing sizes k = 1, 2, ...
    # We know the answer is likely small (<= 5)
    for k in range(1, 6): 
        if time.time() > deadline:
            break

        def dfs(idx, count, current_sum, current_set):
            nonlocal best_set
            if time.time() > deadline:
                return False
            if count == k:
                if current_sum == target:
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
            # Since sorted descending, the top (k-count) are divisors[idx : idx + (k-count)]
            # But we might skip some. So max possible = sum of all remaining
            # Actually, tighter bound: sum of the largest (k-count) available items
            # These are divisors[idx : idx + (k-count)]
            max_possible_add = 0
            for j in range(idx, min(idx + (k - count), len(divisors))):
                max_possible_add += divisors[j]
            
            if current_sum + max_possible_add < target:
                return False
            
            # Min possible additional sum = sum of smallest (k-count) remaining
            # Smallest are at the end.
            # But we are iterating from start.
            # We can check if current_sum + sum of smallest (k-count) > target
            # Smallest (k-count) items in divisors[idx:] are divisors[len(divisors)-(k-count) : len(divisors)]
            # Wait, we must pick from divisors[idx:].
            # The smallest (k-count) items in divisors[idx:] are the last (k-count) items of the whole list?
            # No, they are the last (k-count) items of the sublist divisors[idx:].
            # i.e., divisors[len(divisors) - (k-count) : len(divisors)] is NOT correct if idx > 0.
            # The smallest (k-count) items available are divisors[len(divisors) - (k-count) : len(divisors)]
            # ONLY IF idx <= len(divisors) - (k-count).
            # If idx > len(divisors) - (k-count), then we MUST pick some larger ones.
            
            min_possible_add = 0
            # The smallest (k-count) items in divisors[idx:]
            # They are divisors[len(divisors) - (k-count) : len(divisors)] if those indices are >= idx
            # Otherwise, we pick from the end of divisors[idx:]
            # Actually, the smallest (k-count) items in divisors[idx:] are simply the last (k-count) elements of divisors[idx:]
            # which are divisors[len(divisors) - (k-count) : len(divisors)] if len(divisors) - (k-count) >= idx
            # else divisors[idx : idx + (k-count)]? No.
            
            # Let's just use the fact that divisors are sorted descending.
            # The smallest (k-count) items in divisors[idx:] are divisors[len(divisors) - (k-count) : len(divisors)]
            # provided that len(divisors) - (k-count) >= idx.
            # If not, we don't have enough small items, so we must pick larger ones.
            # In that case, the min sum is the sum of the smallest available, which are the last (k-count) of the available list.
            # Available list is divisors[idx:].
            # Its last (k-count) elements are divisors[len(divisors) - (k-count) : len(divisors)]
            # Wait, if idx is large, say idx=50, and k-count=5, len=60.
            # Available: 50..59. Smallest 5 are 55..59.
            # So min_possible_add = sum(divisors[len(divisors) - (k-count) : len(divisors)])
            # This is correct as long as len(divisors) - (k-count) >= idx.
            # If len(divisors) - (k-count) < idx, then we don't have (k-count) items in the "small" tail.
            # But we always have at least (k-count) items available (checked above).
            # So the smallest (k-count) items are always the last (k-count) items of the entire list?
            # No. If idx=5, k-count=2, len=60.
            # Available: 5..59. Smallest 2 are 58, 59.
            # Which are divisors[58], divisors[59].
            # So yes, the smallest (k-count) items in divisors[idx:] are always divisors[len(divisors)-(k-count) : len(divisors)]
            # because the list is sorted descending, so the smallest are at the end.
            # And since we have at least (k-count) items, the end of the list is within the available range?
            # Yes, because available is a suffix of the list.
            
            min_possible_add = sum(divisors[len(divisors) - (k-count) : len(divisors)])
            
            if current_sum + min_possible_add > target:
                return False

            # Branch 1: Include divisors[idx]
            current_set.append(divisors[idx])
            if dfs(idx + 1, count + 1, current_sum + divisors[idx], current_set):
                current_set.pop()
                return True
            current_set.pop()

            # Branch 2: Exclude divisors[idx]
            if (len(divisors) - (idx + 1)) >= (k - count):
                if dfs(idx + 1, count, current_sum, current_set):
                    return True

            return False

        # Start DFS for this k
        if dfs(0, 0, 0, []):
            break

    if best_set is None:
        # Fallback if no solution found in k=1..5 (should not happen)
        # Try simple greedy or return empty
        best_set = []
        
    return best_set
