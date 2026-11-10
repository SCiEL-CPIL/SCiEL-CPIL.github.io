"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _check_mean_violation(set_vals, a):
    """Check if element a is the arithmetic mean of any subset of >=2 OTHER elements."""
    others = [x for x in set_vals if x != a]
    n = len(others)
    if n < 2:
        return False
    
    # For each subset size k >= 2, check if there exists a subset of 'others' of size k
    # whose sum equals k*a
    # Use DP: dp[k][s] = True if we can pick k elements from others summing to s
    # But s can be large. Max sum = sum(others). For k elements, max sum = k * max(others)
    # We only care if sum == k*a for some k.
    
    # More efficient: for each k from 2 to n, check if k*a is achievable as sum of k elements
    # from others. Use bitset DP.
    
    max_sum = sum(others)
    # dp[k] is a bitset where bit s is set if we can pick k elements summing to s
    # Use list of integers as bitsets
    
    # dp[k] will be a Python int used as bitset
    # dp[k] has bit s set if sum of k elements = s is achievable
    
    # Initialize: dp[0] = 1 (bit 0 set, sum 0 with 0 elements)
    # dp[k] for k > 0 starts as 0
    
    # We process each element in others
    # For each element x, we update dp in reverse order of k
    # dp[k] |= (dp[k-1] << x)
    
    # But we need to check for each k if bit (k*a) is set in dp[k]
    
    # Optimization: we only need to track sums up to max_sum
    # Also, for a given k, the maximum possible sum is k * max(others)
    # and minimum is k * min(others)
    
    # Let's use a list of ints for dp, indexed by k
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0 set
    
    for x in others:
        # Update in reverse to avoid using same element twice
        for k in range(min(len(dp)-1, n), 0, -1):
            # dp[k] |= (dp[k-1] << x)
            # But we need to mask to avoid huge numbers
            # Max relevant sum is max_sum
            dp[k] |= (dp[k-1] << x)
    
    # Now check for each k >= 2 if dp[k] has bit (k*a) set
    for k in range(2, n + 1):
        target = k * a
        if target <= max_sum:
            if (dp[k] >> target) & 1:
                return True
    
    return False

def _is_valid_set(set_vals):
    """Check if no element is the mean of any subset of >=2 other elements."""
    for a in set_vals:
        if _check_mean_violation(set_vals, a):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    # Strategy: greedy with multiple passes and different orderings
    # Try to build a set of up to 48 elements from [1, 512]
    
    best_set = []
    
    # First, try a simple greedy from 1 to 512
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        candidate = selected + [x]
        # Check if adding x creates a violation
        # Only need to check if x is mean of others, or if any existing element becomes mean
        # Actually, adding x can only create new violations involving x
        # Check: is x the mean of any subset of >=2 elements from selected?
        # Also: could any existing element a in selected become the mean of a subset including x?
        
        # Check if x is mean of subset of selected
        valid = True
        if len(selected) >= 2:
            if _check_mean_violation(candidate, x):
                valid = False
        
        # Check if any a in selected is now mean of subset including x
        if valid:
            for a in selected:
                # Check if a is mean of some subset of candidate that includes x
                # This is trickier. We need to check if there's a subset S of selected such that
                # (sum(S) + x) / (len(S) + 1) = a
                # i.e., sum(S) + x = a * (len(S) + 1)
                # i.e., sum(S) = a * (len(S) + 1) - x
                # For each k = len(S) + 1 >= 2, we need sum of (k-1) elements from selected = a*k - x
                # But this is complex. Let's just do the full check for now.
                pass
        
        # For efficiency, let's just do the full check on the candidate
        # But that's O(n^2) per addition. With n up to 48, it's manageable.
        
        if valid:
            # Full check to be safe
            if _is_valid_set(candidate):
                selected = candidate
        
        if len(selected) >= 48:
            break
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Try different orderings to potentially get larger sets
    # Randomized greedy
    import random
    random.seed(42)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        order = list(range(1, 513))
        random.shuffle(order)
        
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            candidate = selected + [x]
            if _is_valid_set(candidate):
                selected = candidate
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Try to improve by swapping elements
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = False
            current = best_set[:]
            # Try removing each element and adding a different one
            for i, a in enumerate(current):
                if time.time() > deadline:
                    break
                remaining = current[:i] + current[i+1:]
                # Try adding each x not in remaining
                for x in range(1, 513):
                    if x in remaining:
                        continue
                    candidate = remaining + [x]
                    if _is_valid_set(candidate):
                        # Check if this gives a different set (we want to keep size same)
                        # Actually, we're just trying to find a valid set of same size
                        # This doesn't improve size, so skip for now
                        break
                # Try to add an extra element
                # Find an x not in current that can be added
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current + [x]
                    if len(candidate) > 48:
                        continue
                    if _is_valid_set(candidate):
                        best_set = candidate
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_set[:48]
# EVOLVE-BLOCK-END
