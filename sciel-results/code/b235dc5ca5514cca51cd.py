# EVOLVE-BLOCK-START
import time
import random

def _can_add_bitset(S, cand):
    """
    Check if cand can be added to S without violating the mean condition.
    Uses bitset DP for subset sum with cardinality constraints.
    S is a list of existing elements (not including cand).
    Returns True if cand can be safely added.
    """
    n = len(S)
    if n < 1:
        return True
    
    # We need to check two things:
    # 1. cand is not the mean of any subset of S with size >= 2
    #    i.e., no subset T of S with |T| >= 2 and sum(T) = |T| * cand
    # 2. No element a in S is the mean of any subset of (S ∪ {cand}) with size >= 2
    #    that includes cand.
    #    i.e., no subset T of S with |T| >= 1 such that (sum(T) + cand) / (|T| + 1) = a
    #    => sum(T) = a * (|T| + 1) - cand = a * |T| + a - cand
    
    # DP: dp[k] is a bitset where bit s is set if there's a subset of S of size k with sum s.
    # Max sum: n * 512. For n up to 48, max sum ~ 24576.
    # We'll use Python integers as bitsets.
    
    max_sum = n * 512
    # dp[k] is an integer where bit i represents sum i
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for x in S:
        # Update in reverse to avoid using x multiple times
        for k in range(n, 0, -1):
            # dp[k] |= dp[k-1] << x
            dp[k] |= (dp[k - 1] << x)
    
    # Check 1: Is cand the mean of any subset of S with size >= 2?
    # Need: exists k >= 2 such that bit (k * cand) is set in dp[k]
    for k in range(2, n + 1):
        target = k * cand
        if target <= max_sum and (dp[k] >> target) & 1:
            return False
    
    # Check 2: For each a in S, is a the mean of a subset including cand?
    # Need: exists a in S, exists k >= 1 such that bit (a*k + a - cand) is set in dp[k]
    # Note: a*k + a - cand must be >= 0 and <= max_sum
    for a in S:
        for k in range(1, n + 1):
            target = a * k + a - cand
            if 0 <= target <= max_sum and (dp[k] >> target) & 1:
                return False
    
    return True

def _conflict_count(x, all_numbers):
    """
    Count how many 3-element arithmetic progressions x could be part of.
    x is the mean: a, x, b where a + b = 2x, a != x, b != x, a != b.
    Also count where x is an endpoint: x, y, z where 2y = x + z, etc.
    Actually, the condition is about x being the MEAN of a subset.
    For 2-element subsets: x is mean of {a, b} iff a + b = 2x.
    For larger subsets, it's more complex, but 2-element conflicts are the most common.
    Let's count pairs (a, b) in all_numbers such that a + b = 2x, a != b, a != x, b != x.
    """
    count = 0
    for a in all_numbers:
        if a == x:
            continue
        b = 2 * x - a
        if b != x and b != a and b in all_numbers:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    best_set = []
    best_len = 0
    
    # Precompute conflict counts for all numbers 1..512
    all_numbers = set(range(1, 513))
    conflict_counts = {}
    for x in range(1, 513):
        conflict_counts[x] = _conflict_count(x, all_numbers)
    
    # Sort candidates by conflict count (fewer conflicts first)
    # This is a heuristic: numbers that are less likely to form arithmetic progressions
    # are "safer" to add early.
    candidates_sorted = sorted(range(1, 513), key=lambda x: conflict_counts[x])
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # For the first restart, use the sorted order.
        # For subsequent restarts, use random shuffles.
        if restart == 0:
            candidates = list(candidates_sorted)
        else:
            candidates = list(range(1, 513))
            random.shuffle(candidates)
        
        current_set = []
        current_set_set = set()
        
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            
            if cand in current_set_set:
                continue
            
            if _can_add_bitset(current_set, cand):
                current_set.append(cand)
                current_set_set.add(cand)
        
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
            
            if best_len >= 48:
                return best_set
    
    # Try to add any remaining numbers to best_set
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        # Sort remaining by conflict count
        remaining.sort(key=lambda x: conflict_counts[x])
        
        for cand in remaining:
            if time.time() > deadline - 2.0:
                break
            if _can_add_bitset(best_set, cand):
                best_set.append(cand)
    
    return best_set

# EVOLVE-BLOCK-END
