import time
import random

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality."""
    n = len(elements)
    if n == 0:
        return [0] * 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << e)
    return dp

def _is_valid_mean_free_with_dp(A, x):
    n = len(A)
    
    # 1. Check if x is the mean of any subset of A of size k >= 2
    dp_A = _build_dp(A)
    for k in range(2, n + 1):
        target = k * x
        if (dp_A[k] >> target) & 1:
            return False
            
    # 2. Check if any a in A is the mean of any subset of (A U {x}) \ {a}
    # Subset must contain x.
    # Sum(T) + x = k * a, where T subset of A \ {a}, |T| = k - 1 = j.
    # Sum(T) = j * a + a - x.
    
    # Pre-sort A to help with bounds? 
    # A is a list. Let's make a sorted copy for bounds calculation.
    A_sorted = sorted(A)
    
    for a in A:
        others = [v for v in A if v != a]
        if len(others) < 1:
            continue
        
        others_sorted = sorted(others)
        
        # Check if we need to build DP
        # We need to check for j in 1..len(others)
        # Target = j * a + a - x
        # Min sum of j elements from others: sum(others_sorted[:j])
        # Max sum of j elements from others: sum(others_sorted[-j:])
        
        # If for all j, target is outside [min_j, max_j], skip DP.
        needs_dp = False
        for j in range(1, len(others) + 1):
            target = j * a + a - x
            if target < 0:
                continue
            min_sum = sum(others_sorted[:j])
            max_sum = sum(others_sorted[-j:])
            if min_sum <= target <= max_sum:
                needs_dp = True
                break
        
        if not needs_dp:
            continue
            
        dp_others = _build_dp(others)
        
        for j in range(1, len(others) + 1):
            target = j * a + a - x
            if target < 0:
                continue
            # Optional: re-check bounds here? No, we already checked.
            if (dp_others[j] >> target) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = []
    best_len = 0
    
    # Strategy 1: Descending Greedy
    # This is often very effective for this type of problem.
    candidates_desc = list(range(512, 0, -1))
    current_set = []
    for x in candidates_desc:
        if time.time() > deadline:
            break
        if _is_valid_mean_free_with_dp(current_set, x):
            current_set.append(x)
            
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
        
    if best_len >= 48:
        return best_set

    # Strategy 2: Random Greedy with bias towards larger numbers
    # We will do multiple restarts.
    # To save time, we limit the number of restarts based on remaining time.
    # Each restart takes ~T seconds.
    # Let's estimate T from the first run? Or just try a fixed number.
    
    # Let's try 20 more restarts with random orders, but biased.
    for i in range(20):
        if time.time() > deadline:
            break
        
        # Create a biased random order: larger numbers have higher probability
        # Or just shuffle and sort by a random key?
        # Let's just shuffle, but maybe weight the selection.
        # Simpler: Just shuffle.
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if time.time() > deadline:
                break
            if _is_valid_mean_free_with_dp(current_set, x):
                current_set.append(x)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
        if best_len >= 48:
            break
            
    return best_set
