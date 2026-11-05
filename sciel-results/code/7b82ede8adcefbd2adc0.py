import time

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a branch-and-bound approach with exact integer arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as (numerator, denominator) pairs for exact arithmetic
    # 1/d is represented as (1, d)
    # We will maintain current sum as (num, den)
    
    best_sum_num = 0
    best_sum_den = 1
    best_indices = []
    
    # To compare sums a/b and c/d without fractions: a*d vs c*b
    # We want to maximize the sum (minimize deficit).
    
    # Helper to check if a/b < 1 (always true for our sums if we prune correctly, 
    # but we need to ensure strict inequality for the final answer)
    # Actually, we just track the max sum < 1.
    
    # Precompute suffix sums of reciprocals to help pruning?
    # Sum of 1/d for d in denominators[i:]
    # This can be used to bound the potential increase in sum.
    # However, simple branch and bound on the current sum vs best sum is often sufficient.
    # Let's use a tighter bound: if current_sum + sum_remaining_max < best_sum, prune.
    
    # Precompute exact fractional suffix sums? That's complex.
    # Instead, use a floating point approximation for pruning bounds, 
    # but exact arithmetic for the actual sum comparison.
    
    suffix_float_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_float_sums[i] = suffix_float_sums[i + 1] + 1.0 / denominators[i]
        
    # DFS Branch and Bound
    # State: index i, current_num, current_den, current_indices
    # We iterate through items. At each step, we can include or exclude.
    
    # Optimization: Sort denominators? 
    # The set is fixed. Order matters for pruning efficiency.
    # Including smaller denominators first increases the sum faster, potentially hitting 1 sooner 
    # or allowing better pruning if we are trying to get close to 1.
    # However, we want the sum < 1. 
    # Let's stick to the natural order [2, 3, ..., 48].
    
    # We will search for the maximum sum < 1.
    
    def dfs(i, cur_num, cur_den, indices):
        nonlocal best_sum_num, best_sum_den, best_indices
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: 
        # 1. If current sum >= 1, we can't use this path (and adding more will only increase it).
        #    Since we only add positive terms, if cur_num/cur_den >= 1, we prune.
        if cur_num >= cur_den:
            return
            
        # 2. Check if this current state is better than the best found so far.
        #    Compare cur_num/cur_den with best_sum_num/best_sum_den
        #    cur_num * best_sum_den > best_sum_num * cur_den
        if cur_num * best_sum_den > best_sum_num * cur_den:
            best_sum_num = cur_num
            best_sum_den = cur_den
            best_indices = indices[:]
            
        # If we've considered all items, return
        if i == n:
            return
            
        # Pruning: Even if we add all remaining items, can we beat the current best?
        # Current sum + suffix_float_sums[i] <= best_sum (float approx)
        # This is a necessary condition. If it fails, we prune.
        # Note: This is a loose bound because float sum of remaining might be slightly 
        # different from exact, but it's safe for pruning if we add a small epsilon buffer 
        # or just rely on the fact that if float says "no", exact almost certainly says "no" 
        # unless we are extremely close. Given the discrete nature, this is safe.
        # To be perfectly safe, we can skip this float prune or make it very conservative.
        # Let's use it conservatively: only prune if clearly below.
        current_float = cur_num / cur_den
        max_possible_float = current_float + suffix_float_sums[i]
        best_float = best_sum_num / best_sum_den
        
        if max_possible_float < best_float - 1e-15:
            return

        # Option 1: Include item i
        d = denominators[i]
        new_num = cur_num * d + cur_den
        new_den = cur_den * d
        dfs(i + 1, new_num, new_den, indices + [i])
        
        # Option 2: Exclude item i
        dfs(i + 1, cur_num, cur_den, indices)

    # Start DFS
    dfs(0, 0, 1, [])
    
    # The problem asks for indices. 
    # If no solution found (should not happen as [0] gives 1/2 < 1), return [0]
    if not best_indices:
        return [0]
        
    return best_indices

if __name__ == "__main__":
    # Self-test
    res = construct()
    # print(res)
    pass
