"""Rewritten with fast subset-sum DP for validity checks and multi-restart local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_fast(selected):
    """
    Check if no element in 'selected' is the arithmetic mean of any subset
    of at least two OTHER elements.
    
    Uses subset-sum DP: for each element a, check if any target T = k*a
    (for k >= 2) can be formed by a subset of the other elements with exactly k elements.
    """
    n = len(selected)
    if n < 3:
        return True
    
    for i, a in enumerate(selected):
        # We need to check if any subset of 'others' (size >= 2) has sum = k*a
        # where k is the size of that subset. So sum = size * a.
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        
        # For each possible subset size k (2..m), check if sum k*a is achievable
        # with exactly k elements from 'others'.
        # DP: dp[s][c] = True if sum s can be formed with exactly c elements
        # Max sum = sum(others)
        max_sum = sum(others)
        
        # We only need to check targets k*a for k in 2..m
        # But we can do a single DP over all possible counts and sums.
        # dp[c] is a bitset of achievable sums using exactly c elements.
        # Use integer bitsets for speed.
        
        # dp[c] is an integer where bit s is set if sum s is achievable with c elements
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in others:
            # Update in reverse to avoid using x multiple times
            for c in range(m, 0, -1):
                dp[c] |= dp[c-1] << x
        
        # Now check: for each k in 2..m, is bit (k*a) set in dp[k]?
        for k in range(2, m + 1):
            target = k * a
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False
    
    return True

def _is_valid_fast_incremental(selected, candidate):
    """
    Check if adding 'candidate' to 'selected' maintains validity.
    We need to check:
    1. 'candidate' is not the mean of any subset of >=2 elements from 'selected'.
    2. No element in 'selected' becomes the mean of a subset that includes 'candidate'.
    """
    # Check 1: candidate not mean of subset of selected
    a = candidate
    others = selected
    m = len(others)
    if m >= 2:
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for c in range(m, 0, -1):
                dp[c] |= dp[c-1] << x
        for k in range(2, m + 1):
            target = k * a
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False
    
    # Check 2: for each element in selected, it's not the mean of a subset including candidate
    # For element a in selected, we check subsets of (selected - {a}) + {candidate}
    # that have size >= 2 and sum = k*a.
    # Subsets can be:
    #   - entirely from selected - {a} (already valid by assumption)
    #   - include candidate: subset of (selected - {a}) of size k-1, sum = k*a - candidate
    # So for each a in selected, for each k >= 2:
    #   check if k*a - candidate can be formed by exactly k-1 elements from selected - {a}
    
    for a in selected:
        others_without_a = [x for x in selected if x != a]
        m2 = len(others_without_a)
        if m2 == 0:
            continue
        max_sum2 = sum(others_without_a)
        
        dp2 = [0] * (m2 + 1)
        dp2[0] = 1
        for x in others_without_a:
            for c in range(m2, 0, -1):
                dp2[c] |= dp2[c-1] << x
        
        # Subsets including candidate have size k (>=2), so k-1 elements from others_without_a
        # sum of those k-1 elements = k*a - candidate
        # k-1 ranges from 1 to m2
        for j in range(1, m2 + 1):  # j = k-1
            k = j + 1
            target = k * a - candidate
            if target < 0 or target > max_sum2:
                continue
            if (dp2[j] >> target) & 1:
                return False
    
    return True

def _greedy_construct(deadline):
    """Greedy construction with fast validity checks."""
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if _is_valid_fast_incremental(selected, x):
            selected.append(x)
        if len(selected) >= 48:
            break
    return selected

def _local_search(initial, deadline):
    """
    Local search: try to add more elements or swap to improve cardinality.
    """
    selected = list(initial)
    current_size = len(selected)
    if current_size >= 48:
        return selected
    
    # Try to add elements
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if x in selected:
            continue
        if _is_valid_fast_incremental(selected, x):
            selected.append(x)
            current_size += 1
            if current_size >= 48:
                break
    
    # Try swaps: remove one, add another
    if current_size < 48 and time.time() < deadline:
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            for x in range(1, 513):
                if x in selected:
                    continue
                # Try replacing selected[i] with x
                new_selected = selected[:i] + selected[i+1:]
                if _is_valid_fast_incremental(new_selected, x):
                    new_selected.append(x)
                    new_selected.sort()
                    # Verify full validity
                    if _is_valid_fast(new_selected):
                        selected = new_selected
                        break
            if len(selected) >= 48:
                break
    
    return selected

def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # 85s out of 90s limit
    
    # Phase 1: Greedy construction
    selected = _greedy_construct(total_deadline)
    
    if len(selected) >= 48:
        return selected
    
    # Phase 2: Local search to try to add more elements
    if time.time() < total_deadline:
        selected = _local_search(selected, total_deadline)
    
    if len(selected) >= 48:
        return selected
    
    # Phase 3: Multiple restarts with different orderings
    best = selected
    remaining = total_deadline - time.time()
    
    if remaining > 5:
        # Try a few different greedy orderings
        attempts = 0
        while time.time() < total_deadline - 2:
            attempts += 1
            # Different ordering: start from different ranges or random
            if attempts % 3 == 1:
                # Try from high to low
                candidates = list(range(512, 0, -1))
            elif attempts % 3 == 2:
                # Try evens first
                candidates = list(range(2, 513, 2)) + list(range(1, 513, 2))
            else:
                # Random shuffle
                candidates = list(range(1, 513))
                random.shuffle(candidates)
            
            sel = []
            for x in candidates:
                if time.time() > total_deadline - 1:
                    break
                if _is_valid_fast_incremental(sel, x):
                    sel.append(x)
                if len(sel) >= 48:
                    break
            
            if len(sel) > len(best):
                best = sel
                if len(best) >= 48:
                    break
            
            # Local search on this result
            if len(best) < 48 and time.time() < total_deadline - 2:
                best = _local_search(best, total_deadline - 1)
                if len(best) >= 48:
                    break
    
    return best

# EVOLVE-BLOCK-END
