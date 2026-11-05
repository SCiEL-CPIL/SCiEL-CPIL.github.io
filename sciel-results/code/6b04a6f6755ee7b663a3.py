"""Rewritten program: minimize the largest distinct factor > 12 with product 12!."""
# EVOLVE-BLOCK-START
import math
import time
import heapq

TARGET = math.factorial(12)
MIN_F = 13
MAX_FACTORS = 12
DEADLINE = 75.0  # safety margin before 90s kill


def _can_complete(remaining, min_f, count_left):
    """Check if remaining can be factored into count_left distinct ints >= min_f."""
    if count_left == 0:
        return remaining == 1
    if remaining < 1:
        return False
    # Lower bound: min_f * (min_f+1) * ... * (min_f+count_left-1)
    lb = 1
    for i in range(count_left):
        lb *= (min_f + i)
    if lb > remaining:
        return False
    # Upper bound: remaining must be >= min_f (smallest single factor)
    if remaining < min_f:
        return False
    return True


def _find_best():
    """Min-heap search over (max_factor, factors_tuple) to find minimal max factor."""
    start = time.time()
    # Heap entries: (max_factor, factors_tuple)
    # We want to find the smallest max_factor such that a valid decomposition exists.
    # Start with the trivial solution: [TARGET] with max=TARGET
    heap = [(TARGET, (TARGET,))]
    best_max = TARGET
    visited = set()  # To avoid redundant states: (remaining, min_f, count_left)
    # Actually, let's track visited states to prune
    # State: (remaining, min_factor, count_used)
    seen = set()
    
    while heap:
        if time.time() - start > DEADLINE:
            break
        max_f, factors = heapq.heappop(heap)
        
        if max_f >= best_max:
            # No point continuing if we can't beat current best
            continue
        
        # Check if this is a complete solution
        if len(factors) == 1:
            # Single factor = TARGET, which is valid
            if factors[0] > MIN_F:
                best_max = min(best_max, max_f)
            continue
        
        # Try to split one of the factors into two smaller distinct factors
        # We want to reduce the max, so try splitting the largest factor
        # But we need to be smart: try all factors, split into a*b where a*b=factor
        # and a, b > MIN_F, a != b, a != all other factors, b != all other factors
        
        for idx in range(len(factors)):
            f = factors[idx]
            if f <= MIN_F:
                continue
            # Try to split f into a * b where a < b, both > MIN_F, distinct from all factors
            # a must be at most sqrt(f)
            a_max = int(math.isqrt(f))
            for a in range(MIN_F, a_max + 1):
                if f % a != 0:
                    continue
                b = f // a
                if b <= a:
                    break
                if b <= MIN_F:
                    continue
                # Check distinctness
                if a in factors or b in factors:
                    continue
                # Check that a and b are distinct from each other (a < b ensures this)
                # New factors list
                new_factors = list(factors[:idx]) + [a, b] + list(factors[idx+1:])
                if len(new_factors) > MAX_FACTORS:
                    continue
                new_max = max(new_factors)
                if new_max >= best_max:
                    continue
                
                # State key for dedup
                key = (tuple(sorted(new_factors)),)
                if key in seen:
                    continue
                seen.add(key)
                
                heapq.heappush(heap, (new_max, tuple(sorted(new_factors, reverse=True))))
    
    return best_max, None  # We only need the max value for scoring


def construct():
    """Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor."""
    start = time.time()
    
    # Quick check: is [TARGET] valid? Yes, TARGET > 12, it's a single distinct integer.
    # But we want to minimize the max. Let's search.
    
    # Use a smarter approach: binary search on the max factor M,
    # and check if 12! can be written as a product of distinct integers in [13, M].
    
    def can_factor_with_max(M, remaining, min_f, count_left, used_set):
        """Check if remaining can be factored into count_left distinct ints in [min_f, M]."""
        if count_left == 0:
            return remaining == 1
        if remaining < 1:
            return False
        if min_f > M:
            return False
        
        # Lower bound check
        lb = 1
        for i in range(count_left):
            lb *= (min_f + i)
        if lb > remaining:
            return False
        
        # Upper bound: the largest possible product with count_left numbers <= M
        # is M * (M-1) * ... but they must be >= min_f and distinct
        # Actually just check if remaining can be achieved
        
        # Try to pick a factor
        # The factor must be in [min_f, M], divide remaining, not in used_set
        # To minimize max, we want smaller factors first? Or larger?
        # For feasibility, try larger factors first to reduce remaining quickly
        
        # Pruning: if count_left == 1, remaining must be in [min_f, M] and not in used_set
        if count_left == 1:
            if min_f <= remaining <= M and remaining not in used_set:
                return True
            return False
        
        # Try factors from M down to min_f (larger first for pruning)
        # But also, the factor must divide remaining
        for f in range(min(M, remaining), min_f - 1, -1):
            if remaining % f != 0:
                continue
            if f in used_set:
                continue
            if f > M:
                continue
            new_remaining = remaining // f
            new_used = used_set | {f}
            if can_factor_with_max(M, new_remaining, max(min_f, f + 1), count_left - 1, new_used):
                return True
        return False
    
    # Binary search on M
    lo = MIN_F
    hi = TARGET  # Trivial solution
    
    # First, verify the trivial solution works
    # M = TARGET: can we factor TARGET into 1 distinct int > 12? Yes: [TARGET]
    
    # Let's search for the minimum M
    # But the search space is large. Let's use a different approach:
    # Directly search for decompositions and track the best max.
    
    # Better approach: use the heap-based search but with better pruning
    # Let me implement a cleaner version
    
    best = TARGET  # The trivial solution
    best_factors = [TARGET]
    
    # DFS with pruning, trying to find solutions with smaller max
    # We'll explore in order of increasing max factor
    
    # Actually, let me use a direct approach: enumerate possible sets of factors
    # This is complex. Let me use a recursive search with memoization.
    
    def dfs(remaining, min_f, count_left, current_factors, used_set):
        """Recursively find a valid factorization, tracking the best max."""
        nonlocal best, best_factors
        if time.time() - start > DEADLINE:
            return
        
        if count_left == 0:
            if remaining == 1:
                cur_max = max(current_factors) if current_factors else 0
                if cur_max < best:
                    best = cur_max
                    best_factors = list(current_factors)
            return
        
        if remaining < 1:
            return
        
        # Lower bound: minimum product of count_left distinct ints >= min_f
        lb = 1
        for i in range(count_left):
            lb *= (min_f + i)
        if lb > remaining:
            return
        
        # Upper bound: remaining must be >= min_f
        if remaining < min_f:
            return
        
        # If count_left == 1, remaining must be a valid factor
        if count_left == 1:
            if min_f <= remaining and remaining not in used_set:
                cur_max = max(current_factors + [remaining])
                if cur_max < best:
                    best = cur_max
                    best_factors = list(current_factors) + [remaining]
            return
        
        # Try factors from min_f upward
        # To minimize the max, we should try smaller factors first
        # But we also need to ensure the remaining can be completed
        for f in range(min_f, int(math.isqrt(remaining)) + 1):
            if remaining % f != 0:
                continue
            if f in used_set:
                continue
            if f > best:  # If f alone exceeds best, no need to continue
                break
            
            new_remaining = remaining // f
            new_count = count_left - 1
            new_min_f = f + 1
            new_used = used_set | {f}
            
            # Pruning: can the remaining be factored into new_count distinct ints >= new_min_f?
            lb2 = 1
            for i in range(new_count):
                lb2 *= (new_min_f + i)
            if lb2 > new_remaining:
                continue
            
            # Also, the max of any completion will be at least max(f, new_min_f)
            # If that already exceeds best, skip
            if max(f, new_min_f) >= best:
                continue
            
            dfs(new_remaining, new_min_f, new_count, current_factors + [f], new_used)
            
            # Also try: what if f is the last factor?
            # That's handled by count_left == 1 case above
        
        # Also try: remaining itself as a single factor if count_left allows
        # This is handled when we reach count_left == 1
    
    dfs(TARGET, MIN_F, MAX_FACTORS, [], set())
    
    # If we found a better solution, return it; otherwise return the trivial one
    if best < TARGET:
        return best_factors
    else:
        return [TARGET]


# EVOLVE-BLOCK-END
