import time
import random
import sys

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _subset_sum_exists_non_empty(w, target):
    """
    Check if there is a non-empty subset of w that sums to target.
    w: list of integers
    target: integer
    Returns True if such a subset exists.
    """
    if not w:
        return False
    
    # Quick check for single element
    if target in w:
        return True

    # Calculate bounds for offset
    min_sum = sum(x for x in w if x < 0)
    max_sum = sum(x for x in w if x > 0)
    
    # If target is out of bounds, return False
    if target < min_sum or target > max_sum:
        return False

    offset = -min_sum
    size = max_sum - min_sum + 1
    mask = (1 << size) - 1
    
    # dp bitset, bit i is set if sum (i - offset) is achievable
    # We want non-empty subsets.
    # Initialize dp to 0.
    dp = 0
    
    for val in w:
        if val >= 0:
            # Shift left by val
            new_dp = dp << val
            # Also add the subset consisting of just 'val'
            # The bit for sum 'val' is at index (val - min_sum)
            # But we are doing incremental DP.
            # Standard 0/1 knapsack style:
            # dp_new = dp_old | (dp_old << val)
            # But we need to include the single element 'val'.
            # If we start with dp=0, after first element val1:
            # dp = 0 | (0 << val1) = 0. This misses {val1}.
            # So we should initialize or handle the "start" case.
            # Better: dp represents sums of non-empty subsets of processed elements.
            # When adding val:
            # New sums = Old sums + val, and {val}.
            # So: dp_new = (dp << val) | (1 << (val + offset))
            # Wait, the index for sum S is S + offset.
            # So bit for val is at val + offset.
            dp = (dp << val) | (1 << (val + offset))
        else:
            # val < 0
            # Shift right by -val
            # dp_new = (dp >> -val) | (1 << (val + offset))
            dp = (dp >> -val) | (1 << (val + offset))
        
        # Mask to keep within bounds
        dp &= mask
        
        # Early exit: check if target is achievable
        target_bit = target + offset
        if (dp >> target_bit) & 1:
            return True

    return False

def _is_mean_of_subset(target, others):
    """
    Check if target is the mean of any subset of others with size >= 2.
    others: list of integers
    target: integer
    Returns True if violation exists.
    """
    if len(others) < 2:
        return False
    
    # Transform: z_i = y_i - target
    # We need a subset S of z with |S| >= 2 and sum(S) = 0.
    zs = [y - target for y in others]
    
    # Check for pairs (size 2) quickly
    # If any z_i == 0, then we need another z_j such that z_i + z_j = 0?
    # No, if z_i == 0, subset {z_i, z_j} sums to z_j. We need sum 0.
    # So if z_i == 0, we need another z_j == 0? Or just {z_i, z_j} where z_j=0?
    # Actually, if z_i == 0, the subset {z_i} has sum 0 but size 1.
    # We need size >= 2.
    # So if there are two zeros, we have a violation.
    # If there is one zero, we need a subset of the rest summing to 0.
    
    # Let's just use the general DP for non-empty subset summing to 0.
    # But we must ensure size >= 2.
    # My _subset_sum_exists_non_empty checks for ANY non-empty subset.
    # If it finds a subset of size 1 summing to 0, that means z_i = 0.
    # If z_i = 0, is that a violation?
    # Original condition: target is mean of subset S of others, |S|>=2.
    # sum(S) = |S| * target.
    # sum(y - target) = 0.
    # If S has size 1, y - target = 0 => y = target.
    # But y is in others, so y != target (distinct elements).
    # So z_i cannot be 0?
    # Wait, others are distinct from target?
    # In our context, `others` are elements from the set A, and target is x.
    # x is not in A. So y != x. So z_i != 0.
    # So any non-empty subset summing to 0 has size >= 2?
    # Not necessarily. If z_i + z_j = 0, size 2.
    # Can a single z_i sum to 0? Only if z_i=0, which implies y=target, impossible.
    # So yes, any non-empty subset of zs summing to 0 has size >= 2.
    
    return _subset_sum_exists_non_empty(zs, 0)

def _check_add(x, A):
    """
    Check if adding x to A creates any violation.
    A: list of selected integers
    x: candidate integer
    Returns True if valid (no violation), False if violation.
    """
    # 1. Check if x is the mean of any subset of A (size >= 2)
    if _is_mean_of_subset(x, A):
        return False
    
    # 2. Check if any a in A is the mean of any subset of (A \ {a}) U {x} (size >= 2)
    for a in A:
        # Others for a are (A \ {a}) U {x}
        # We need to check if a is mean of subset S of others, |S|>=2.
        # sum(S) = |S| * a
        # Let S = {x} U S', where S' subset of A \ {a}
        # x + sum(S') = (1 + |S'|) * a
        # sum(S') - |S'| * a = a - x
        # Let w_i = b_i - a for b_i in A \ {a}
        # We need a non-empty subset of w summing to a - x.
        # Note: S' can be empty?
        # If S' is empty, S = {x}, size 1. Not allowed.
        # So S' must be non-empty.
        
        others = [b for b in A if b != a]
        w = [b - a for b in others]
        target = a - x
        
        if _subset_sum_exists_non_empty(w, target):
            return False
            
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate orders
    orders = []
    
    # 1. Random orders
    rng = random.Random(42)
    for _ in range(100):
        if time.time() - start_time > _DEADLINE - 5:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # 2. Structured orders
    orders.append(sorted(candidates))
    orders.append(sorted(candidates, reverse=True))
    orders.append(list(range(256, 513)) + list(range(1, 256)))
    orders.append(list(range(1, 513, 2)) + list(range(2, 513, 2)))

    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            if _check_add(x, selected):
                selected.append(x)
                
                if len(selected) > len(best_set):
                    best_set = selected[:]
                    
                if len(selected) >= 48:
                    break
                    
    return best_set

# EVOLVE-BLOCK-END
