import math
from itertools import combinations

# Precompute LCM of 1..24
def lcm(a, b):
    return a * b // math.gcd(a, b)

L = 1
for i in range(1, 25):
    L = lcm(L, i)

# v_k = L // k
v = [0] * 25
for k in range(1, 25):
    v[k] = L // k

def subset_sum_exists(nums, target):
    """
    Returns True if any subset of nums sums to target.
    Uses meet-in-the-middle.
    """
    if target < 0:
        return False
    if target == 0:
        return True
    
    n = len(nums)
    if n == 0:
        return target == 0
    
    # If the sum of all numbers is less than target, impossible
    if sum(nums) < target:
        return False
    
    # If the smallest number is greater than target, impossible (since all positive)
    if min(nums) > target:
        return False

    mid = n // 2
    left = nums[:mid]
    right = nums[mid:]
    
    # Generate all subset sums for left
    sums_left = [0]
    for num in left:
        # Add num to all existing sums
        sums_left += [s + num for s in sums_left]
        
    # Generate all subset sums for right
    sums_right = [0]
    for num in right:
        sums_right += [s + num for s in sums_right]
        
    # Sort right sums for binary search
    sums_right.sort()
    
    # Check if any s in sums_left has (target - s) in sums_right
    import bisect
    for s in sums_left:
        diff = target - s
        if diff < 0:
            continue
        # Binary search for diff in sums_right
        idx = bisect.bisect_left(sums_right, diff)
        if idx < len(sums_right) and sums_right[idx] == diff:
            return True
            
    return False

def is_valid(S):
    """
    Checks if S is a valid reciprocal-subsum-free set.
    S is a list of integers.
    """
    for x in S:
        target = v[x]
        # Get values of other elements
        others = [v[y] for y in S if y != x]
        if subset_sum_exists(others, target):
            return False
    return True

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe
    
    S = []
    # Try adding elements from largest to smallest
    for k in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        S_candidate = S + [k]
        # Optimization: If adding k doesn't change validity for existing elements?
        # It might, because k is now available as a summand for other targets.
        # So we must check full validity.
        
        if is_valid(S_candidate):
            S = S_candidate
            
    # If S is empty (shouldn't happen as 24 is always valid), return [24]
    if not S:
        return [24]
        
    return S
