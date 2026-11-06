"""Rewritten program for large reciprocal-subsum-free sets."""
import time
from math import gcd
from functools import reduce

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_subsum_free(S, k):
    """
    Check if 1/k can be expressed as a sum of reciprocals of a non-empty
    subset of S.
    Returns True if it is free (cannot be formed), False otherwise.
    """
    if not S:
        return True
    
    # Convert to integers: 1/x = L / (L*x)
    # We want to find subset T of S such that sum(1/x for x in T) == 1/k
    # This is equivalent to: sum(L / (L*x)) == L / (L*k)
    # Multiply by L: sum(1 / (x/L? No))
    # Let L = LCM(S, k).
    # 1/x = (L/x) / L.
    # So sum(1/x) = 1/k  <=>  sum(L/x) = L/k.
    # Let a_x = L/x for x in S.
    # Let target = L/k.
    # We want to know if target is in the set of subset sums of {a_x}.
    
    # Note: S might contain k? 
    # The problem says: "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements."
    # So if we are checking if we can add k to S, we check if 1/k is a subsum of S.
    # If k is already in S, we don't call this for k.
    # In the greedy loop, k is not in S.
    
    # Calculate LCM of all elements in S and k.
    # S contains integers.
    L = k
    for x in S:
        L = _lcm(L, x)
        if L > 10**12: # Overflow guard, though for N=24 LCM is bounded by LCM(1..24) which is ~ 2.6e10
             # Actually LCM(1..24) is 5354228880. So L is at most that.
             pass
    
    target = L // k
    
    # Values to sum
    vals = [L // x for x in S]
    
    # Sort values in descending order for better pruning in DFS?
    # Or ascending?
    # If we sort descending, we try large contributions first.
    # Pruning: if current_sum + sum(remaining) < target, prune.
    # If current_sum > target, prune.
    
    n = len(vals)
    if n == 0:
        return True
        
    # Quick check: if sum(vals) < target, return True (free)
    total_sum = sum(vals)
    if total_sum < target:
        return True
        
    # If any val == target, return False (not free)
    # If any val > target, it can't be part of a valid subset (since all positive), so we can ignore it?
    # Yes, if val > target, including it makes sum > target. So we can filter it out.
    vals = [v for v in vals if v <= target]
    
    if not vals:
        return True
        
    if target in vals:
        return False
        
    # Sort descending to find solutions faster?
    # Or ascending?
    # Let's sort descending.
    vals.sort(reverse=True)
    
    # Precompute suffix sums
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + vals[i]
        
    # DFS
    found = False
    
    def dfs(i, current_sum):
        nonlocal found
        if found:
            return
        if i == n:
            if current_sum == target:
                found = True
            return
            
        # Prune 1: if current_sum > target
        if current_sum > target:
            return
            
        # Prune 2: if current_sum + suffix[i] < target
        if current_sum + suffix[i] < target:
            return
            
        # Option 1: Include vals[i]
        # If including it leads to a valid sum, we are done.
        # Check if current_sum + vals[i] == target
        if current_sum + vals[i] == target:
            found = True
            return
        elif current_sum + vals[i] < target:
            dfs(i + 1, current_sum + vals[i])
            if found:
                return
        # If current_sum + vals[i] > target, we can't include it.
        
        # Option 2: Exclude vals[i]
        dfs(i + 1, current_sum)
        
    dfs(0, 0)
    
    return not found

def construct():
    # Greedy strategy: Add elements in descending order.
    # This is the standard approach for this type of problem.
    # We want to maximize cardinality.
    # Adding larger numbers (smaller reciprocals) first is generally safer
    # because they contribute less to the sums, making it harder to accidentally
    # form a sum for a smaller number later?
    # Actually, if we add small numbers first, their reciprocals are large.
    # It is easy to form sums for even smaller numbers.
    # Example: 1/2 + 1/3 = 5/6. 1/6 is 1/6.
    # If we have {2, 3}, can we add 6?
    # 1/6 vs subsets of {1/2, 1/3}.
    # 1/2 != 1/6. 1/3 != 1/6. 1/2+1/3 = 5/6 != 1/6.
    # So {2, 3, 6} is valid?
    # Check 2: 1/2 vs {1/3, 1/6}. 1/3+1/6 = 1/2. YES.
    # So {2, 3, 6} is INVALID. 1/2 is a subsum of {3, 6}.
    
    # So the order matters.
    # If we add 6 first, then 3, then 2.
    # S={6}. Add 3: 1/3 vs {1/6}. 1/6 != 1/3. OK. S={6,3}.
    # Add 2: 1/2 vs {1/6, 1/3}. 1/6+1/3 = 1/2. INVALID.
    
    # So {2,3,6} is bad.
    # What is a good set?
    # The seed returns 13..24.
    # Let's stick to the greedy descending order.
    
    S = []
    for k in range(24, 0, -1):
        if _time_remaining() < 5:
            break
        if _is_subsum_free(S, k):
            S.append(k)
            
    if not S:
        return list(range(13, 25))
        
    return sorted(S)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
