import sys
sys.setrecursionlimit(10000)

def construct():
    # 1. Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Sort to ensure we pick in increasing order
    smooth.sort()
    n_smooth = len(smooth)
    
    best_rep = {}
    targets = set(range(1, 129))
    
    # We search for k=1, 2, ...
    # Max k is bounded by 30 per problem statement, but likely much smaller.
    # We break early if all targets are found.
    
    # Helper to check if x is incomparable with all in current_set
    def is_incomparable(x, current_set):
        for y in current_set:
            if x % y == 0 or y % x == 0:
                return False
        return True

    # Recursive generator for antichains of size k
    # We store results in best_rep
    def backtrack(start_idx, current_set, current_sum, k):
        if len(current_set) == k:
            if current_sum <= 128:
                if current_sum not in best_rep:
                    best_rep[current_sum] = list(current_set)
            return
        
        # Pruning: if we have picked k items, we are done.
        # We need to pick (k - len(current_set)) more items.
        # The smallest possible sum for the remaining items is the sum of the next (k-len) smallest available numbers.
        # But we don't know which are available due to incomparability.
        # Simple bound: If current_sum > 128, return.
        if current_sum > 128:
            return
            
        # If we can't pick enough items left
        remaining_needed = k - len(current_set)
        if n_smooth - start_idx < remaining_needed:
            return
            
        # Heuristic bound on sum:
        # The maximum sum we can add is if we pick the largest remaining numbers?
        # No, we want to know if we can reach <= 128.
        # Since we are summing positive numbers, if current_sum > 128, we pruned.
        # We also need to ensure we don't overshoot 128 with the minimum possible additions?
        # Actually, we just need to find ANY representation.
        
        for i in range(start_idx, n_smooth):
            x = smooth[i]
            # If adding x exceeds 128, and all subsequent numbers are larger, we can break?
            # Yes, because smooth is sorted.
            if current_sum + x > 128:
                # Since we need to pick more numbers (if len < k), and they are positive,
                # the sum will only increase.
                # Wait, if len(current_set) < k, we MUST pick more.
                # So if current_sum + x > 128, any superset including x will have sum > 128.
                # So we can break the loop entirely for this level?
                # No, we can just skip x and any larger x.
                break
                
            if is_incomparable(x, current_set):
                current_set.append(x)
                backtrack(i + 1, current_set, current_sum + x, k)
                current_set.pop()

    for k in range(1, 31):
        if len(best_rep) == 128:
            break
        # Optimization: if we have found all targets, stop.
        # Also, if k is large, the number of combinations explodes.
        # But the antichain constraint prunes heavily.
        backtrack(0, [], 0, k)
        
    # Verify all targets are found
    missing = [n for n in range(1, 129) if n not in best_rep]
    if missing:
        # If some are missing, it might be because k was too small or logic error.
        # Given the problem statement, they should exist.
        # If not, we raise an error to signal failure (score 0).
        raise ValueError(f"Missing targets: {missing}")

    answers = [best_rep[n] for n in range(1, 129)]
    return answers
