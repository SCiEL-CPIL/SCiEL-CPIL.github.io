"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _is_valid_addition(selected, x):
    """Check if adding x to selected keeps the set valid."""
    S = selected
    n = len(S)
    
    # Check if x is the mean of any subset of S of size >= 2
    # Need: subset T of S, |T|>=2, sum(T) = |T| * x
    max_sum = sum(S)
    for k in range(2, n + 1):
        target = k * x
        if target > max_sum:
            break
        # Check if we can form sum = target using k elements from S
        # Use bitset DP: dp[s] = min number of elements to achieve sum s, or we track achievable sums per size
        # For efficiency, use a set of achievable (sum, count) pairs, but that's expensive
        # Alternative: for each k, check subset sum with exactly k elements
        # Since S is small (up to 48), and values up to 512, max sum ~ 48*512 = 24576
        # Use a dictionary: dp[count] = set of sums
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        for val in S:
            for c in range(n, 0, -1):
                for s in dp[c - 1]:
                    dp[c].add(s + val)
        if target in dp[k]:
            return False
    
    # Check if any a in S is the mean of a subset of S' = S ∪ {x} that includes x
    # For each a in S: need subset T of (S \ {a}) ∪ {x} with |T| >= 2 and sum(T) = |T| * a
    # T must include x (otherwise it was already valid in S)
    # So T = {x} ∪ U where U is a subset of S \ {a}, |U| >= 1
    # sum(U) + x = (|U| + 1) * a
    # => sum(U) = (|U| + 1) * a - x
    for a in S:
        others = [y for y in S if y != a]
        m = len(others)
        max_sum_others = sum(others)
        for k_u in range(1, m + 1):  # k_u = |U|, so total subset size = k_u + 1 >= 2
            target = (k_u + 1) * a - x
            if target < 0 or target > max_sum_others:
                continue
            # Check if we can form sum = target using k_u elements from others
            dp = [set() for _ in range(m + 1)]
            dp[0].add(0)
            for val in others:
                for c in range(m, 0, -1):
                    for s in dp[c - 1]:
                        dp[c].add(s + val)
            if target in dp[k_u]:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # 80s deadline, 10s safety margin
    
    selected = []
    for x in range(1, 513):
        if time.time() - start_time > deadline:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
            if len(selected) >= 48:
                break
    
    return selected
# EVOLVE-BLOCK-END
