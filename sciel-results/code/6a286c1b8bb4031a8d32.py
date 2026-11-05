"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = 55.0  # seconds, safety margin before 60s limit


def _build_smooth_pool(n_max=512, max_prime=5):
    """Build a pool of smooth numbers (products of small primes) up to n_max."""
    pool = set()
    # Generate numbers of form 2^a * 3^b * 5^c * 7^d <= n_max
    a = 0
    while (1 << a) <= n_max:
        b = 0
        while (1 << a) * (3 ** b) <= n_max:
            c = 0
            while (1 << a) * (3 ** b) * (5 ** c) <= n_max:
                d = 0
                while (1 << a) * (3 ** b) * (5 ** c) * (7 ** d) <= n_max:
                    pool.add((1 << a) * (3 ** b) * (5 ** c) * (7 ** d))
                    d += 1
                c += 1
            b += 1
        a += 1
    return sorted(pool)


def _check_violation_bitset(candidate, a_idx, max_k=None):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Uses bitset DP for subset-sum with exactly k elements.
    
    For element a, others = candidate without a.
    Need: exists subset S of others, |S|>=2, such that sum(S) = |S|*a.
    Equivalent: exists subset S of {y-a for y in others}, |S|>=2, sum(S)=0.
    
    Bitset approach: dp[k] is an integer where bit s is set if sum s is achievable
    with exactly k elements. We offset sums by OFFSET to handle negatives.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    if max_k is None:
        max_k = n
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # Quick pair check
    zset = set()
    for z in zs:
        if -z in zset and -z != z:
            return True
        if -z == z:
            # Need two distinct elements with value 0
            # Count zeros
            pass
        zset.add(z)
    # More careful pair check
    for i in range(n):
        for j in range(i+1, n):
            if zs[i] + zs[j] == 0:
                return True
    
    # Determine range of possible sums
    # Max absolute sum of any subset of size k from zs
    abs_zs = sorted([abs(z) for z in zs], reverse=True)
    # For k elements, max sum magnitude is sum of top k absolute values
    max_abs_sum = 0
    for k in range(3, max_k + 1):
        if k <= n:
            s = sum(abs_zs[:k])
            if s > max_abs_sum:
                max_abs_sum = s
    
    if max_abs_sum == 0:
        return False
    
    OFFSET = max_abs_sum
    # Bitset: bit i represents sum = i - OFFSET
    # We need to track exactly k elements
    
    # dp[k] is a Python int acting as bitset
    # Bit position p means sum = p - OFFSET is achievable with exactly k elements
    # We need bit OFFSET to be set (sum = 0)
    
    # For efficiency, limit k to min(max_k, n)
    max_k_eff = min(max_k, n)
    
    dp = [0] * (max_k_eff + 1)
    dp[0] = 1 << OFFSET  # sum 0 with 0 elements
    
    for z in zs:
        # Update backwards
        for k in range(max_k_eff, 0, -1):
            if dp[k-1]:
                # Shift dp[k-1] by z
                if z >= 0:
                    dp[k] |= (dp[k-1] << z)
                else:
                    dp[k] |= (dp[k-1] >> (-z))
    
    # Check if sum 0 is achievable with exactly k elements for any k >= 3
    target_bit = 1 << OFFSET
    for k in range(3, max_k_eff + 1):
        if dp[k] & target_bit:
            return True
    
    return False


def _check_violation_full(candidate, a_idx):
    """Full check using set-based DP (slower but handles larger ranges)."""
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    zs = [y - a for y in others]
    
    # Pair check
    for i in range(n):
        for j in range(i+1, n):
            if zs[i] + zs[j] == 0:
                return True
    
    # Set-based DP for k >= 3
    dp = [set() for _ in range(n+1)]
    dp[0].add(0)
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1]:
                for s in dp[k-1]:
                    dp[k].add(s + z)
    
    for k in range(3, n+1):
        if 0 in dp[k]:
            return True
    
    return False


def _full_check(candidate, max_k=None):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_violation_bitset(candidate, i, max_k):
            return True
    return False


def _full_check_sets(candidate):
    """Check all elements using set-based DP (more thorough)."""
    for i in range(len(candidate)):
        if _check_violation_full(candidate, i):
            return True
    return False


def construct():
    start_time = time.time()
    
    # Build candidate pool: smooth numbers (2^a * 3^b * 5^c * 7^d)
    smooth_pool = _build_smooth_pool(512, 7)
    
    # Also include all numbers for completeness in later phases
    all_nums = list(range(1, 513))
    
    best_set = []
    
    # Phase 1: Greedy from smooth pool (most promising for density)
    orders = [
        smooth_pool,                                          # smooth numbers
        sorted(smooth_pool, reverse=True),                    # smooth descending
        [x for x in all_nums if x % 2 != 0],                 # odds
        [x for x in all_nums if x % 2 == 0],                 # evens
        list(range(1, 513)),                                  # increasing
        list(range(512, 0, -1)),                              # decreasing
    ]
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 8:
            break
        
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 8:
                break
            
            candidate = selected + [x]
            new_idx = len(selected)
            
            # Check if new element x is mean of any subset of selected
            if len(selected) >= 2:
                if _check_violation_bitset(candidate, new_idx, max_k=6):
                    continue
            
            # Check if any existing element becomes a mean due to x
            # For small sets, full check
            if len(candidate) <= 14:
                if _full_check(candidate, max_k=6):
                    continue
            else:
                # Check new element more thoroughly
                if _check_violation_bitset(candidate, new_idx, max_k=8):
                    continue
                
                # Spot-check existing elements
                check_indices = [0, len(selected)//2, len(selected)-1]
                violation_found = False
                for idx in check_indices:
                    if idx != new_idx:
                        if _check_violation_bitset(candidate, idx, max_k=6):
                            violation_found = True
                            break
                
                if violation_found:
                    continue
                
                # Thorough check if time allows
                if time.time() - start_time < _DEADLINE - 15 and len(candidate) <= 22:
                    if _full_check(candidate, max_k=8):
                        continue
            
            selected.append(x)
            
            if len(selected) >= 48:
                break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Phase 2: Try to add more elements from remaining numbers
    if len(best_set) < 48 and time.time() - start_time < _DEADLINE - 5:
        used = set(best_set)
        remaining = [x for x in all_nums if x not in used]
        
        improved = True
        while improved and time.time() - start_time < _DEADLINE - 5:
            improved = False
            for x in remaining:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                candidate = best_set + [x]
                new_idx = len(best_set)
                
                if _check_violation_bitset(candidate, new_idx, max_k=6):
                    continue
                
                if len(candidate) <= 20:
                    if _full_check(candidate, max_k=6):
                        continue
                else:
                    if _check_violation_bitset(candidate, new_idx, max_k=8):
                        continue
                    # Spot check
                    for idx in [0, len(best_set)//2, len(best_set)-1]:
                        if _check_violation_bitset(candidate, idx, max_k=6):
                            break
                    else:
                        if _full_check(candidate, max_k=6):
                            continue
                        else:
                            best_set.append(x)
                            remaining.remove(x)
                            improved = True
                            if len(best_set) >= 48:
                                break
                            break
    
    # Final verification
    if best_set:
        if _full_check_sets(best_set):
            # Fallback: return a known valid set
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
