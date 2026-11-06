"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    target = math.factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (evaluator kills at 90s)
    
    # Precompute all divisors of target
    all_divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            all_divisors.add(i)
            all_divisors.add(target // i)
    
    # Sort divisors greater than 12
    valid_divisors = sorted([d for d in all_divisors if d > 12])
    divisor_set = set(valid_divisors)
    
    # Index for fast lookup: value -> position in sorted list
    div_index = {d: i for i, d in enumerate(valid_divisors)}
    
    best = [target]
    best_max = target
    
    # Precompute: for each divisor d > 12, which divisors of (target//d) are also > 12 and > d?
    # This helps with pruning. But let's use a smarter DFS with memoization on (remaining, min_factor_index)
    
    # Better approach: DFS choosing factors in increasing order from valid_divisors
    # State: (remaining_product, min_allowed_index) where min_allowed_index is index into valid_divisors
    # We want to minimize the largest factor
    
    # Use a priority queue / best-first search? Or iterative deepening on max factor?
    # Since we want to minimize the largest factor, we can binary search on the answer.
    
    # Can we achieve a decomposition with all factors <= X and > 12, distinct?
    def can_decompose(remaining, min_idx, max_factor_limit, used_set):
        """Check if remaining can be factored into distinct divisors > 12, all <= max_factor_limit,
           with each factor >= valid_divisors[min_idx]."""
        if time.time() > deadline:
            return None  # timeout
        if remaining == 1:
            return []
        
        # Try factors from min_idx upward, but only those <= max_factor_limit and dividing remaining
        # Also, the factor must be > 12 (guaranteed by valid_divisors)
        for idx in range(min_idx, len(valid_divisors)):
            f = valid_divisors[idx]
            if f > max_factor_limit:
                break
            if f > remaining:
                break
            if remaining % f != 0:
                continue
            if f in used_set:
                continue
            
            used_set.add(f)
            result = can_decompose(remaining // f, idx + 1, max_factor_limit, used_set)
            if result is not None:
                result.append(f)
                used_set.remove(f)
                return result
            used_set.remove(f)
        
        return None
    
    # Binary search on the maximum factor
    # Lower bound: smallest possible max factor. At least 13.
    # Upper bound: target (just [target])
    
    # But we also need at least one factor > 12, so min possible max is at least the smallest divisor > 12
    # Let's find a good initial upper bound by greedy or by trying small values
    
    # Actually, let's just iterate from small max_factor_limit upward
    # The answer is likely not too large. Let's try from 13 upward.
    
    # But can_decompose with a set is expensive. Let's use a different approach.
    
    # DFS with pruning, but only considering valid divisors, and using the fact that we want to minimize max.
    # We'll do DFS in order of increasing factors, and prune if current factor >= best_max.
    
    def dfs_optimized(remaining, min_idx, current_factors):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if len(current_factors) > 0:
                max_f = current_factors[-1]  # since factors are in increasing order
                if max_f < best_max:
                    best_max = max_f
                    best = list(current_factors)
            return
        
        # If remaining is prime or has no valid divisors > min_idx, we can't proceed
        # unless remaining itself is a valid factor > 12 and > last factor
        # Actually, remaining must be expressible as product of distinct factors from valid_divisors[min_idx:]
        
        # Quick check: if remaining is in divisor_set and remaining > valid_divisors[min_idx - 1] (or min_idx == 0):
        # then we could take remaining as the last factor, but only if it's > 12 and not already used
        # But we need to ensure it's distinct from current_factors. Since we're going in increasing order,
        # and remaining > all previous factors (because product of previous factors < remaining * previous factors... not necessarily)
        # Actually, if remaining is a valid divisor and remaining > current_factors[-1] if current_factors is non-empty:
        if remaining in divisor_set:
            if len(current_factors) == 0 or remaining > current_factors[-1]:
                # We can take remaining as the last factor
                current_factors.append(remaining)
                max_f = remaining
                if max_f < best_max:
                    best_max = max_f
                    best = list(current_factors)
                current_factors.pop()
                # Don't return; there might be better decompositions with more factors
                # But if remaining >= best_max, we already recorded it, and no further factor can reduce max below best_max
                # Actually, adding more factors would make the max at least remaining (since remaining is one factor),
                # so if remaining >= best_max, no improvement possible from this branch
                if remaining >= best_max:
                    return
        
        # Try each valid divisor f >= valid_divisors[min_idx] that divides remaining
        # and f < best_max (pruning)
        # Also f must be > 12 (guaranteed) and distinct (ensured by increasing order)
        
        for idx in range(min_idx, len(valid_divisors)):
            f = valid_divisors[idx]
            if f >= best_max:
                break
            if f > remaining:
                break
            if remaining % f != 0:
                continue
            
            current_factors.append(f)
            dfs_optimized(remaining // f, idx + 1, current_factors)
            current_factors.pop()
        
        # Additional pruning: if the smallest possible product of distinct factors from valid_divisors[min_idx:]
        # that multiply to remaining requires a factor >= best_max, we can stop.
        # This is complex, so we rely on the f >= best_max break.
    
    dfs_optimized(target, 0, [])
    
    return best
# EVOLVE-BLOCK-END
