"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    target = factorial(12)
    best_max = float('inf')
    best_factors = None
    
    # Precompute divisors of target that are > 12
    # We only need divisors that could potentially be part of a solution with max < current best_max
    # But best_max changes, so we precompute all valid divisors > 12
    divisors = []
    limit = int(target**0.5) + 1
    for i in range(13, limit):
        if target % i == 0:
            divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    divisors.sort()
    
    # Helper to get min product of k distinct integers > 12 not in used
    def get_min_product(k, used):
        count = 0
        prod = 1
        i = 13
        while count < k:
            if i not in used:
                prod *= i
                count += 1
            i += 1
        return prod

    def dfs_k(remaining, used, current_factors, current_max, k_left):
        nonlocal best_max, best_factors, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if k_left == 0:
            if remaining == 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        if remaining == 1:
            return
        
        # If current max is already >= best_max, no point in continuing
        if current_max >= best_max:
            return
            
        # Pruning 1: Minimum possible product of k_left distinct factors > 12 not in used
        min_prod = get_min_product(k_left, used)
        if remaining < min_prod:
            return
        
        # Pruning 2: The largest factor in the remaining k_left factors must be at least 
        # remaining ** (1/k_left). If this is >= best_max, we can't improve.
        if remaining ** (1.0 / k_left) >= best_max:
            return
            
        # Determine valid divisors for this step
        # d must divide remaining
        # d > 12
        # d not in used
        # d < best_max
        # Also, d must be such that remaining // d can be formed by k_left - 1 factors
        # each < best_max.
        # This implies remaining // d < best_max ** (k_left - 1)
        # So d > remaining / (best_max ** (k_left - 1))
        
        lower_bound_d = 13
        if k_left > 1:
            if best_max < float('inf'):
                lower_bound_d = max(lower_bound_d, int(remaining / (best_max ** (k_left - 1))) + 1)
        
        candidates = []
        # We iterate through divisors. To speed up, we can skip those < lower_bound_d
        # and stop if d >= best_max
        for d in divisors:
            if d < lower_bound_d:
                continue
            if d >= best_max:
                break
            if remaining % d != 0:
                continue
            if d in used:
                continue
            
            # Additional pruning: 
            # The remaining product rem = remaining // d
            # We need k_left - 1 factors from rem.
            # The min product of k_left - 1 distinct factors > 12 not in (used U {d}) must be <= rem
            # And the max factor in the subproblem must be < best_max
            # This is handled by the recursive call, but we can add a quick check.
            if k_left > 1:
                # If rem is too large, it might force a factor >= best_max
                # Specifically, if rem >= best_max ** (k_left - 1), it's *possible* but 
                # the geometric mean argument suggests it's unlikely to be better if 
                # we are already close to best_max.
                # Actually, the condition `remaining ** (1.0 / k_left) >= best_max` at the top
                # already covers the global geometric mean.
                # Let's just rely on the recursive call for now, it's safer.
                pass
            
            candidates.append(d)
            
        # Sort candidates to try promising ones first
        # To minimize the maximum factor, we should try to keep the factors balanced.
        # The ideal value for d is close to remaining ** (1/k_left)
        if candidates:
            target_d = remaining ** (1.0 / k_left)
            candidates.sort(key=lambda d: abs(d - target_d))
        
        for d in candidates:
            new_used = used | {d}
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue
            dfs_k(remaining // d, new_used, current_factors + [d], new_max, k_left - 1)

    # Search for k from 8 down to 1
    # k=8 is the maximum possible number of factors
    # Min product for k=8 is 13*...*20 ~ 1.67e8 < 12!
    # Min product for k=9 is 13*...*21 ~ 3.5e9 > 12!
    # So k is in [1, 8]
    
    for k in range(8, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        dfs_k(target, set(), [], 0, k)
        
    if best_factors is None:
        # Fallback: return a valid solution, even if not optimal
        # 12! = 13 * (12! // 13)
        # 12! // 13 is an integer > 12
        return [13, target // 13]
        
    return best_factors
# EVOLVE-BLOCK-END
