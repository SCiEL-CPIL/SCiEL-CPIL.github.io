"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def _verify(denoms):
    num, den = 0, 1
    for d in denoms:
        num, den = num * d + den, den * d
        g = gcd(num, den)
        if g:
            num //= g
            den //= g
    return num == den

def _dfs(k, start, remain_num, remain_den, chosen, solution, deadline):
    # remain = remain_num / remain_den
    # need sum of 1/d for d >= start, exactly k terms
    if remain_num == 0:
        if k == 0:
            solution.append(list(chosen))
            return True
        return False
    if k == 0:
        return False
    if time.time() > deadline:
        return False
    
    # Lower bound on next denominator:
    # 1/d <= remain/k  =>  d >= k/remain = k * remain_den / remain_num
    lower = (k * remain_den + remain_num - 1) // remain_num
    if lower < start:
        lower = start
    
    # Upper bound: we need to be able to fill the rest
    # The remaining k-1 terms must sum to (remain - 1/d)
    # Each of the next terms is at most 1/(d+1), etc.
    # Actually, let's just bound by: if 1/lower > remain, no solution.
    if 1.0 / lower > remain_num / remain_den + 1e-12:
        return False
    
    # Also, the sum of the largest possible k terms starting from lower must be >= remain
    # This is a loose check, but helpful
    # Max sum = sum_{i=0}^{k-1} 1/(lower + i)
    # If this is less than remain, no solution.
    max_possible = 0.0
    for i in range(k):
        max_possible += 1.0 / (lower + i)
    if max_possible < remain_num / remain_den - 1e-12:
        return False

    # Search for d in [lower, ...]
    # Upper bound for d:
    # We need remain - 1/d >= sum of smallest possible (k-1) terms?
    # No, the terms don't have to be smallest.
    # But 1/d <= remain => d >= 1/remain. This is already covered by lower?
    # lower = k/remain. Since k>=1, k/remain >= 1/remain.
    # Wait, if k=1, lower = 1/remain.
    # If k>1, lower is larger.
    
    # Let's iterate d from lower upwards.
    # How high can d go?
    # The remaining sum after picking d is remain - 1/d.
    # This must be representable by k-1 terms >= d+1.
    # The maximum sum of k-1 terms >= d+1 is sum_{i=1}^{k-1} 1/(d+i).
    # So remain - 1/d <= sum_{i=1}^{k-1} 1/(d+i)
    # Also, the minimum sum of k-1 terms is... unbounded below (denominators can be huge).
    # But we also need remain - 1/d > 0.
    # And 1/d < remain => d > 1/remain.
    
    # A tighter upper bound for d:
    # The remaining k-1 terms must sum to R' = remain - 1/d.
    # The largest possible sum for k-1 terms starting from d+1 is S_max = sum_{i=1}^{k-1} 1/(d+i).
    # We need R' <= S_max.
    # Also, the smallest possible sum is close to 0.
    # But there's a lower bound on R' too? No, we can always pick huge denominators to make the sum small.
    # However, we are constrained to integer denominators.
    
    # Let's just iterate d. We can break if 1/d becomes too small such that
    # even with k-1 terms of size d+1, we can't reach the remaining sum?
    # Wait, we need R' to be reachable.
    # If R' is very small, we can pick huge denominators.
    # If R' is large, we need small denominators.
    
    # Let's try to bound d from above.
    # If k-1 == 0, then R' must be 0. So remain - 1/d = 0 => d = 1/remain.
    # If k-1 > 0:
    # We know the next denominators will be >= d+1.
    # The maximum sum they can contribute is sum_{i=1}^{k-1} 1/(d+i).
    # So we require remain - 1/d <= sum_{i=1}^{k-1} 1/(d+i).
    # This gives an upper bound on d?
    # As d increases, 1/d decreases, so remain - 1/d increases.
    # The RHS also decreases.
    # So for large d, LHS > RHS.
    # Thus there is an upper bound.
    
    # Let's just iterate and check the condition.
    
    # Optimization: 
    # If k == 1:
    #   1/d = remain => d = remain_den / remain_num.
    #   If integer, return it.
    
    if k == 1:
        if remain_den % remain_num == 0:
            d = remain_den // remain_num
            if d >= lower:
                chosen.append(d)
                solution.append(list(chosen))
                chosen.pop()
                return True
        return False

    # General case
    # We can also use the fact that d must divide remain_den * something?
    # remain - 1/d = (remain_num * d - remain_den) / (remain_den * d)
    # Let R' = (remain_num * d - remain_den) / (remain_den * d)
    # We need to represent R' with k-1 terms.
    
    # Iterate d
    # Upper bound heuristic:
    # If d is too large, the remaining sum R' is close to remain.
    # The max sum of k-1 terms starting from d+1 is approx (k-1)/(d+1).
    # If remain - 1/d > (k-1)/(d+1), it's impossible.
    # remain - 1/d > (k-1)/(d+1)
    # remain > 1/d + (k-1)/(d+1)
    # For large d, RHS is small. If remain > 0, this is always true for large d.
    # Wait, if remain is fixed, and d grows, 1/d -> 0, (k-1)/(d+1) -> 0.
    # So remain > 0 is always > RHS for large d.
    # So this condition doesn't give a tight upper bound on d.
    # 
    # Actually, we need R' to be *at least* the minimum possible sum? No.
    # We need R' to be *at most* the maximum possible sum.
    # Max sum of k-1 distinct terms >= d+1 is sum_{i=1}^{k-1} 1/(d+i).
    # So we need remain - 1/d <= sum_{i=1}^{k-1} 1/(d+i).
    # As d increases, LHS increases (towards remain), RHS decreases.
    # So there is a crossover point.
    
    # Let's just iterate d from lower.
    # To avoid infinite loop, we can stop if d is very large.
    # But if a solution exists, d shouldn't be excessively large unless remain is tiny.
    # However, denominators are bounded by 10,000,000.
    # We can cap d at 10,000,000.
    
    # A better upper bound:
    # If k-1 terms are needed, the smallest they can be is d+1, d+2, ...
    # The largest sum they can make is S_max.
    # If R' > S_max, no solution.
    # We can compute S_max and break if R' > S_max.
    
    d = lower
    max_d = 10000000
    while d <= max_d:
        if d < start:
            d += 1
            continue
        if remain_num * d <= remain_den:
            # 1/d >= remain, so R' <= 0.
            # If R' == 0, we need k-1 terms to sum to 0, impossible.
            # If R' < 0, impossible.
            # So if 1/d >= remain, we must have 1/d == remain and k-1 == 0?
            # But we handled k==1 above.
            # So for k>1, we need 1/d < remain.
            d += 1
            continue
        
        # R' = remain - 1/d > 0
        r_num = remain_num * d - remain_den
        r_den = remain_den * d
        g = gcd(r_num, r_den)
        if g:
            r_num //= g
            r_den //= g
        
        # Check if R' can be represented by k-1 terms starting from d+1
        # Max sum check
        if k - 1 == 1:
            # Need 1/x = R' => x = r_den / r_num
            if r_den % r_num == 0:
                x = r_den // r_num
                if x > d:
                    chosen.append(d)
                    chosen.append(x)
                    solution.append(list(chosen))
                    chosen.pop()
                    chosen.pop()
                    return True
            d += 1
            continue
        
        # For k-1 > 1, check if R' <= sum_{i=1}^{k-1} 1/(d+i)
        # This is a loose check, but helps prune.
        # Actually, let's just recurse. The recursion will handle the bounds.
        # But we should ensure we don't go too deep or too wide.
        
        chosen.append(d)
        if _dfs(k - 1, d + 1, r_num, r_den, chosen, solution, deadline):
            chosen.pop()
            return True
        chosen.pop()
        
        d += 1
        if time.time() > deadline:
            return False
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # We know k=8 is the minimum.
    # Let's search for k=8 first.
    # If not found, try k=9, etc.
    # But we are fairly confident k=8 exists.
    
    # Initial state: sum = 1/1
    for k in range(8, 13):
        solution = []
        # Use DFS
        if _dfs(k, 5, 1, 1, [], solution, deadline):
            if solution:
                return solution[0]
        
        if time.time() > deadline:
            break
            
    # Fallback: return a known valid solution if any found, or a generic one
    # Actually, the task requires returning 1..256 distinct denominators.
    # If we fail to find a small k, we can return a larger set.
    # But we should always return *something* valid if possible.
    # The seed program uses DP on divisors of 2520.
    # Let's use the seed's logic as a fallback if DFS fails or times out.
    
    # Fallback to seed logic
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
