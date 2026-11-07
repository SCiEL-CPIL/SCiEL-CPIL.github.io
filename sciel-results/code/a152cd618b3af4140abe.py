import time
import sys
from itertools import combinations

def _build_dp(arr):
    """Build dp table: dp[k] = bitset of sums achievable with exactly k elements."""
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            if dp[k-1] != 0:
                dp[k] |= (dp[k-1] << x)
    return dp

def _is_valid(arr):
    """
    Checks if the set 'arr' satisfies the condition:
    Sums of subsets with different cardinalities must be disjoint.
    """
    n = len(arr)
    if n == 0:
        return True
    
    dp = _build_dp(arr)
    
    # Check for intersections between different cardinality sum sets
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
        
    return True

def _try_add(arr, x, dp):
    """
    Try adding x to arr with existing dp table.
    Returns (new_dp, valid) or (None, False).
    """
    n = len(arr)
    # Extend dp to n+1 entries
    new_dp = dp[:] + [0]
    # Update from high to low
    for k in range(n + 1, 0, -1):
        if new_dp[k-1] != 0:
            new_dp[k] |= (new_dp[k-1] << x)
    
    # Check validity
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None, False
        acc |= new_dp[k]
    
    return new_dp, True

def _try_swap(arr, i, x, dp):
    """
    Try replacing arr[i] with x. Returns (new_arr, new_dp, valid) or (None, None, False).
    """
    n = len(arr)
    old_val = arr[i]
    if x == old_val:
        return None, None, False
    if x in arr:
        return None, None, False
    
    new_arr = arr[:]
    new_arr[i] = x
    
    # Rebuild dp from scratch (simpler and fast enough for n<=60)
    new_dp = _build_dp(new_arr)
    
    # Check validity
    acc = 0
    for k in range(n + 1):
        if new_dp[k] & acc:
            return None, None, False
        acc |= new_dp[k]
    
    return new_arr, new_dp, True

def _grow_greedy(arr, dp, deadline):
    """Grow arr greedily by trying to add candidates."""
    for x in range(256, 0, -1):
        if time.monotonic() > deadline:
            break
        if x in arr:
            continue
        new_dp, valid = _try_add(arr, x, dp)
        if valid:
            arr.append(x)
            dp = new_dp
    return arr, dp

def _swap_search(arr, dp, deadline, max_rounds=20):
    """Local search: try swaps to enable further growth."""
    n = len(arr)
    for round_num in range(max_rounds):
        if time.monotonic() > deadline:
            break
        
        improved = False
        # Try swapping each element with a candidate not in arr
        for i in range(n):
            if time.monotonic() > deadline:
                break
            for x in range(256, 0, -1):
                if x in arr:
                    continue
                new_arr, new_dp, valid = _try_swap(arr, i, x, dp)
                if valid:
                    # Check if this enables adding a new element
                    # Try to grow after the swap
                    new_arr2, new_dp2 = _grow_greedy(new_arr, new_dp, deadline)
                    if len(new_arr2) > n:
                        arr = new_arr2
                        dp = new_dp2
                        n = len(arr)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # No swap helped; try adding directly
            arr, dp = _grow_greedy(arr, dp, deadline)
            if len(arr) == n:
                break
            n = len(arr)
    
    return arr, dp

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0  # Safety margin before 90s kill
    
    # Start with empty set
    arr = []
    dp = _build_dp(arr)
    
    # Phase 1: Greedy growth
    arr, dp = _grow_greedy(arr, dp, deadline)
    
    # Phase 2: Swap-based local search to escape local optima
    arr, dp = _swap_search(arr, dp, deadline)
    
    # Phase 3: Final greedy pass
    arr, dp = _grow_greedy(arr, dp, deadline)
    
    # Ensure Python ints
    best = [int(v) for v in arr]
    
    # Fallback if something went wrong (should not happen)
    if not best:
        best = [1]
        
    return best
