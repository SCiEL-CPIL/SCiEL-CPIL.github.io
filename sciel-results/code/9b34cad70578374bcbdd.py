"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _check_violation_bitset(candidate, a_idx):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Uses bitset DP for speed.
    
    For element a, others = candidate without a.
    Need: exists subset S of others, |S|>=2, such that sum(S) = |S|*a.
    Equivalent: exists subset S of {y-a for y in others}, |S|>=2, sum(S)=0.
    
    dp[k] is a bitmask where bit s is set if sum s-offset is achievable with k elements.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # Calculate max possible sum magnitude for offset
    # Sum of absolute values is a safe upper bound for the range of subset sums
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False
        
    offset = max_sum
    # Bitset size: 2 * max_sum + 1
    # We need to be careful with memory if max_sum is very large, but n<=512 and vals<=512
    # Max sum of diffs is bounded by n * 512.
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitmask
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # Sum 0 is achievable with 0 elements
    
    for z in zs:
        # Update backwards to avoid using the same element twice
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            dp[k] |= (shifted & mask)
            
            # Early exit: if we can form sum 0 (bit offset) with k >= 2
            if k >= 2 and (dp[k] & (1 << offset)):
                return True

    # Final check for any k >= 2
    for k in range(2, n + 1):
        if dp[k] & (1 << offset):
            return True
            
    return False

def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_violation_bitset(candidate, i):
            return True
    return False

def _try_add_element(current_set, x, deadline):
    """
    Try to add x to current_set. Returns True if valid, False otherwise.
    Only checks if x creates a violation (x is mean of others) or if any existing element 
    becomes a mean due to x. 
    For efficiency, we first check if x is a mean of any subset of current_set.
    Then we check if any element in current_set is a mean of a subset including x.
    """
    if x in current_set:
        return False
    
    candidate = current_set + [x]
    new_idx = len(current_set)
    
    # Check if x is mean of any subset of current_set
    if _check_violation_bitset(candidate, new_idx):
        return False
    
    # Check if any existing element is now a mean involving x
    # This is expensive. We check all existing elements.
    # Optimization: If time is tight, we might skip some, but for correctness we must check.
    # However, adding one element only creates new subsets that include x.
    # So we only need to check if any a in current_set is the mean of a subset S 
    # such that x in S and |S|>=2.
    # This is equivalent to checking if a is the mean of {x} U T where T is a subset of current_set\{a}.
    # sum({x} U T) = |{x} U T| * a
    # x + sum(T) = (1 + |T|) * a
    # sum(T) - |T|*a = a - x
    # Let T' = {t - a for t in T}. Then sum(T') = a - x.
    # We need to find a subset T of current_set\{a} such that sum(t-a) = a-x.
    
    for i in range(len(current_set)):
        a = current_set[i]
        target_sum = a - x
        # We need subset of {t - a for t in current_set if t != a} that sums to target_sum
        # And size of subset T can be 0? No, |S|>=2 means 1+|T|>=2 => |T|>=1.
        # So we need non-empty subset T.
        
        # Quick check: if target_sum is 0, we need sum(T')=0 with |T|>=1.
        # If target_sum != 0, just check subset sum.
        
        # Use a simple DP for subset sum with size constraint?
        # Or just use the bitset check on the modified candidate?
        # _check_violation_bitset(candidate, i) checks if a is mean of ANY subset of others.
        # Since the set was valid before, the only new violations are those involving x.
        # So _check_violation_bitset(candidate, i) is sufficient and correct.
        
        if time.time() > deadline:
            return False # Timeout, assume invalid to be safe or handle outside
        
        if _check_violation_bitset(candidate, i):
            return False
            
    return True

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_set = []
    
    # Phase 1: Greedy construction with multiple random orders
    # We want to get a large base set quickly.
    
    # Try a few deterministic good orders first
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        list(range(1, 513, 2)) + list(range(2, 513, 2)),
        list(range(2, 513, 2)) + list(range(1, 513, 2)),
    ]
    
    # Add some random orders
    for _ in range(10):
        rand_order = list(range(1, 513))
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        current_set = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 10:
                break
                
            # Try to add x
            if _try_add_element(current_set, x, _DEADLINE - 10):
                current_set.append(x)
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                    if len(best_set) >= 48:
                        # Found maximum possible
                        return best_set
        
        # If we have time, try to improve this specific set with local search
        # Local Search: Try to add any remaining number, or swap one out for two?
        # Swapping is complex. Let's just try to add more numbers not in the set.
        
        if time.time() - start_time < _DEADLINE - 5 and len(current_set) < 48:
            remaining = [x for x in range(1, 513) if x not in current_set]
            random.shuffle(remaining)
            
            improved = False
            for x in remaining:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                if _try_add_element(current_set, x, _DEADLINE - 5):
                    current_set.append(x)
                    improved = True
                    if len(current_set) > len(best_set):
                        best_set = current_set[:]
                    if len(best_set) >= 48:
                        return best_set
            
            # If we improved, we might try again? No, just keep going to next order.
            
    # Phase 2: If best_set is not 48, try a more aggressive local search on the best_set found
    if len(best_set) < 48 and time.time() - start_time < _DEADLINE - 5:
        current_best = best_set[:]
        remaining = [x for x in range(1, 513) if x not in current_best]
        
        # Try to add any single element
        added_any = True
        while added_any and time.time() - start_time < _DEADLINE - 5:
            added_any = False
            random.shuffle(remaining)
            for x in remaining:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                if _try_add_element(current_best, x, _DEADLINE - 5):
                    current_best.append(x)
                    remaining.remove(x)
                    added_any = True
                    if len(current_best) > len(best_set):
                        best_set = current_best[:]
                    if len(best_set) >= 48:
                        return best_set
        
        # Try swaps? Remove one, add two?
        # This is harder. Let's stick to additions for now.
        
    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
