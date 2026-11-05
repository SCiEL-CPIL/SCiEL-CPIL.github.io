"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = 55.0  # seconds, safety margin before 60s limit


def _check_violation_bitset(candidate, a_idx):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Uses bitset DP for speed.
    
    For element a, others = candidate without a.
    Need: exists subset S of others, |S|>=2, such that sum(S) = |S|*a.
    Equivalent: exists subset S of {y-a for y in others}, |S|>=2, sum(S)=0.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # Quick pair check first
    for i in range(n):
        for j in range(i + 1, n):
            if zs[i] + zs[j] == 0:
                return True
    
    # Calculate max possible sum magnitude for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False
        
    offset = max_sum
    # Bitset size: 2 * max_sum + 1
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitmask where bit s is set if sum s-offset is achievable with k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # Sum 0 is achievable with 0 elements
    
    for z in zs:
        # Update backwards to avoid using the same element twice
        for k in range(n, 0, -1):
            if dp[k - 1] == 0:
                continue
            if z >= 0:
                shifted = dp[k - 1] << z
            else:
                shifted = dp[k - 1] >> (-z)
            dp[k] |= (shifted & mask)
            
            # Early exit: if we can form sum 0 (bit offset) with k >= 3
            if k >= 3 and (dp[k] & (1 << offset)):
                return True

    # Final check for any k >= 3 (k=2 already checked above)
    for k in range(3, n + 1):
        if dp[k] & (1 << offset):
            return True
            
    return False


def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_violation_bitset(candidate, i):
            return True
    return False


def construct():
    start_time = time.time()
    
    best_set = []
    
    # Try multiple orders to find the largest valid set
    orders = [
        list(range(1, 513)),                          # increasing
        list(range(512, 0, -1)),                      # decreasing
        list(range(256, 513)) + list(range(1, 256)),  # upper then lower
        list(range(1, 513, 2)) + list(range(2, 513, 2)),  # odds then evens
        list(range(2, 513, 2)) + list(range(1, 513, 2)),  # evens then odds
    ]
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            candidate = selected + [x]
            new_idx = len(selected)
            
            # Check if x is mean of any subset of selected
            if _check_violation_bitset(candidate, new_idx):
                continue
            
            # Also need to verify existing elements aren't now means
            # Do full check for small sets, spot check for larger
            if len(candidate) <= 20:
                if _full_check(candidate):
                    continue
            else:
                # Spot check a few existing elements
                check_indices = [0, len(selected) // 2, len(selected) - 1]
                violation_found = False
                for idx in check_indices:
                    if idx != new_idx and _check_violation_bitset(candidate, idx):
                        violation_found = True
                        break
                
                if violation_found:
                    continue
                
                # If we have time, do a more thorough check
                if time.time() - start_time < _DEADLINE - 15:
                    if _full_check(candidate):
                        continue
            
            selected.append(x)
            
            if len(selected) >= 48:
                break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback: return a known valid set
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
