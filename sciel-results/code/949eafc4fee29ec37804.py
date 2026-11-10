"""Rewritten program: Fast bitset-based greedy with optimistic conflict check and exact verification."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512
MAX_TIME = 55.0  # safety margin under 60s

def _exact_check(S):
    """Exact check for validity. Slow but correct."""
    S_list = sorted(S)
    n = len(S_list)
    if n < 3:
        return True
    
    for idx_a, a in enumerate(S_list):
        others = [x for i, x in enumerate(S_list) if i != idx_a]
        m = len(others)
        if m < 2:
            continue
        
        max_sum = sum(others)
        # Use list of sets for reachable sums by size
        reachable = [set() for _ in range(m + 1)]
        reachable[0].add(0)
        
        for x in others:
            for k in range(m - 1, 0, -1):
                if reachable[k - 1]:
                    for s in list(reachable[k - 1]):
                        new_s = s + x
                        if new_s <= max_sum:
                            reachable[k].add(new_s)
        
        for k in range(2, m + 1):
            if k * a in reachable[k]:
                return False
    return True

def _fast_check_add(x, S, dp):
    """
    Optimistic fast check for adding x to S.
    dp[k] is a bitset (int) where bit s is 1 if sum s is achievable by k elements of S.
    Returns True if x can likely be added.
    """
    if not S:
        return True
    
    # 1. Check if x is mean of any subset of S of size >= 2
    # x is mean of subset U (size k) if sum(U) = k*x
    for k in range(2, len(S) + 1):
        target_sum = k * x
        if target_sum > 512 * k: # Max possible sum
            continue
        if (dp[k] >> target_sum) & 1:
            return False
            
    # 2. Check if any a in S becomes mean of a subset including x
    # a is mean of subset V (size j+1, includes x) if sum(V\{x}) + x = (j+1)*a
    # sum(V\{x}) = (j+1)*a - x. Size of V\{x} is j.
    # V\{x} is a subset of S of size j.
    for a in S:
        for j in range(1, len(S) + 1):
            # j is size of subset from S. Total size j+1 >= 2 is satisfied since j>=1.
            target_sum = (j + 1) * a - x
            if target_sum < 0 or target_sum > 512 * j:
                continue
            if (dp[j] >> target_sum) & 1:
                # Conflict found in optimistic check.
                # We accept this as a conflict to be safe, even if it might be due to 'a' being in the subset.
                # This might reject valid candidates, but ensures we don't accept invalid ones easily.
                return False
    return True

def _update_dp(dp, x, new_len):
    """Update dp bitsets after adding x. dp is a list of ints."""
    # Iterate backwards
    for k in range(new_len, 0, -1):
        if k > 1:
            dp[k] |= (dp[k-1] << x)
        else:
            dp[1] |= (1 << x)
    return dp

def _greedy_fast(elements, rng):
    """Greedy construction using fast optimistic checks."""
    selected = []
    dp = [0] * (len(elements) + 1)
    dp[0] = 1 # bit 0 set
    
    remaining = list(elements)
    rng.shuffle(remaining)
    
    for x in remaining:
        if _fast_check_add(x, selected, dp):
            selected.append(x)
            _update_dp(dp, x, len(selected))
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_set = []
    rng = random.Random(42)
    elements = list(range(1, N + 1))
    
    # Try a few random restarts
    while time.time() < deadline - 2.0:
        s = _greedy_fast(elements, rng)
        
        # Exact verification
        if _exact_check(s):
            if len(s) > len(best_set):
                best_set = s
        else:
            # If invalid, try to repair by removing elements?
            # Or just restart. Given time, restart is safer.
            pass
            
        # Try to improve best_set by adding more elements
        if len(best_set) > 0 and time.time() < deadline - 1.0:
            current = list(best_set)
            # Rebuild dp for current
            dp = [0] * (N + 1)
            dp[0] = 1
            for x in current:
                _update_dp(dp, x, len(current))
            
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                # Try adding any element not in current
                for x in elements:
                    if x not in current:
                        if _fast_check_add(x, current, dp):
                            # Verify this specific addition exactly?
                            # For speed, we trust the fast check for addition, 
                            # but we should verify the whole set at the end.
                            current.append(x)
                            _update_dp(dp, x, len(current))
                            improved = True
                            break
                
                if len(current) > len(best_set):
                    # Check if current is exactly valid
                    if _exact_check(current):
                        best_set = current
                    else:
                        # If invalid, we might need to backtrack, but for now we just keep best_set
                        pass

    return best_set
# EVOLVE-BLOCK-END
