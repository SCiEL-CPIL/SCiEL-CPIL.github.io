# EVOLVE-BLOCK-START
import time
import random

def _subset_sum_bitset(arr, target_sum, min_count, max_count):
    """
    Check if there exists a subset of arr with size in [min_count, max_count]
    that sums to target_sum. Uses bitset DP.
    Returns True if such a subset exists.
    """
    n = len(arr)
    if target_sum < 0 or target_sum > n * max(arr) if arr else 0:
        return False
    if min_count > n or max_count < min_count:
        return False
    max_count = min(max_count, n)
    
    # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
    # We only need to track up to target_sum
    dp = [0] * (max_count + 1)
    dp[0] = 1  # bit 0 set: sum 0 with 0 elements
    
    for x in arr:
        # Update in reverse to avoid using x multiple times
        for k in range(min(max_count, n), 0, -1):
            if dp[k-1]:
                dp[k] |= (dp[k-1] << x)
        # Mask to keep only bits up to target_sum
        mask = (1 << (target_sum + 1)) - 1
        for k in range(max_count + 1):
            dp[k] &= mask
    
    # Check if any k in [min_count, max_count] has bit target_sum set
    for k in range(min_count, max_count + 1):
        if dp[k] & (1 << target_sum):
            return True
    return False

def _can_add_fast(S, cand):
    """
    Fast check: can cand be added to S?
    S is a sorted list of distinct ints. cand is not in S.
    Returns True if valid.
    """
    n = len(S)
    if n < 2:
        return True
    
    # Check 1: cand is not mean of any subset of S with size >= 2
    # For k in 2..n: is there a subset of S of size k summing to k*cand?
    for k in range(2, n + 1):
        if _subset_sum_bitset(S, k * cand, k, k):
            return False
    
    # Check 2: No a in S is mean of any subset of S U {cand} with size >= 2 that includes cand
    # For each a in S, for k in 1..n: is there a subset T of S of size k such that
    # sum(T) = a*k + a - cand?
    for a in S:
        for k in range(1, n + 1):
            target = a * k + a - cand
            if target < 0:
                continue
            if _subset_sum_bitset(S, target, k, k):
                return False
    
    return True

def _try_add(S, cand, S_set):
    """Try to add cand to S. Returns True if added."""
    if cand in S_set:
        return False
    if not _can_add_fast(S, cand):
        return False
    S.append(cand)
    S_set.add(cand)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    best_set = []
    best_len = 0
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Structured sampling: start with numbers spread out
        # Use a pattern that avoids arithmetic progressions
        candidates = list(range(1, 513))
        
        # Bias towards numbers that are less likely to form means
        # Use a weighted random selection or structured pattern
        if restart % 3 == 0:
            # Spread out: take every other number, then fill in
            even = [x for x in candidates if x % 2 == 1]
            odd = [x for x in candidates if x % 2 == 0]
            random.shuffle(even)
            random.shuffle(odd)
            candidates = even + odd
        elif restart % 3 == 1:
            # Random shuffle
            random.shuffle(candidates)
        else:
            # Start from high numbers (less likely to be means of smaller numbers)
            candidates = list(range(512, 0, -1))
            random.shuffle(candidates[256:])  # Shuffle lower half
            candidates = candidates[256:] + candidates[:256]
        
        current_set = []
        current_set_set = set()
        
        # Greedy construction
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            if len(current_set) >= 48:
                break
            _try_add(current_set, cand, current_set_set)
        
        # Local search: try to add more elements
        if len(current_set) < 48:
            remaining = [x for x in range(1, 513) if x not in current_set_set]
            random.shuffle(remaining)
            for cand in remaining:
                if time.time() > deadline - 3.0:
                    break
                if len(current_set) >= 48:
                    break
                _try_add(current_set, cand, current_set_set)
        
        # Local search: swap operations
        # Try removing one element and adding two others
        if len(current_set) < 48:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for i in range(len(current_set)):
                    if time.time() > deadline - 2.0:
                        break
                    removed = current_set[i]
                    new_set = current_set[:i] + current_set[i+1:]
                    new_set_set = set(new_set)
                    
                    # Try to add two elements
                    remaining = [x for x in range(1, 513) if x not in new_set_set]
                    random.shuffle(remaining)
                    added = 0
                    for cand in remaining:
                        if added >= 2:
                            break
                        if time.time() > deadline - 2.0:
                            break
                        if _try_add(new_set, cand, new_set_set):
                            added += 1
                    
                    if added > 1 or (added == 1 and len(new_set) > len(current_set)):
                        current_set = new_set
                        current_set_set = new_set_set
                        improved = True
                        break
        
        if len(current_set) > best_len:
            best_set = current_set[:]
            best_len = len(current_set)
            if best_len >= 48:
                return best_set
    
    return best_set

# EVOLVE-BLOCK-END
