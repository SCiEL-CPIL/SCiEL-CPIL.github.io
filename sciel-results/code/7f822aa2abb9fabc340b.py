"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55

    # Attempt 1: Dynamic Programming on small LCMs
    # This is very fast and likely to find the optimal k=8 solution
    # if it exists within the divisor set.
    
    def solve_dp(target_lcm):
        # Find divisors of target_lcm that are >= 5
        candidates = []
        for d in range(5, target_lcm + 1):
            if target_lcm % d == 0:
                candidates.append(d)
        
        if not candidates:
            return None
            
        # dp[s] = min number of terms to sum to s
        # parent[s] = the denominator d that was added to reach s
        INF = float('inf')
        dp = [INF] * (target_lcm + 1)
        parent = [-1] * (target_lcm + 1)
        dp[0] = 0
        
        # 0/1 Knapsack to minimize count
        # We iterate through each candidate denominator once.
        for d in candidates:
            w = target_lcm // d
            # Iterate backwards to ensure each d is used at most once
            for s in range(target_lcm, w - 1, -1):
                if dp[s - w] + 1 < dp[s]:
                    dp[s] = dp[s - w] + 1
                    parent[s] = d
        
        if dp[target_lcm] == INF:
            return None
            
        # Reconstruct the solution
        result = []
        s = target_lcm
        while s > 0:
            d = parent[s]
            if d == -1:
                return None # Should not happen if dp[target] is not INF
            result.append(d)
            s -= target_lcm // d
            
        return result

    # Try small LCMs first
    for lcm in [2520, 27720]:
        if time.time() > deadline:
            break
        result = solve_dp(lcm)
        if result is not None:
            # Verify solution just in case
            # sum(1/d for d in result) == 1
            # len(result) <= 256
            # all distinct, in range
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                if abs(sum(Fraction(1, d) for d in result) - 1) < 1e-9:
                    return result

    # Attempt 2: DFS Fallback
    # This handles cases where the optimal solution uses non-divisors of small LCMs
    def dfs(remaining, min_d, max_terms, current, max_d_limit):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current
        if max_terms == 0:
            return None
        if remaining <= 0:
            return None

        # Pruning: max possible sum with max_terms terms starting from min_d
        # Approximation for speed, or exact Fraction
        # Using float for pruning check is faster, but risky.
        # Let's use Fraction for correctness in the tight loop, but optimize if possible.
        # Given the time limit, Fraction is likely fine for small depths.
        
        # Max possible sum
        max_possible = Fraction(0)
        for i in range(max_terms):
            max_possible += Fraction(1, min_d + i)
        if remaining > max_possible:
            return None

        # Min possible sum
        min_possible = max_terms * Fraction(1, max_d_limit)
        if remaining < min_possible:
            return None

        # Bounds for next denominator d
        # 1/d <= remaining  => d >= 1/remaining
        # Also d >= min_d
        if remaining >= 1:
            lower = min_d
        else:
            # int(1/remaining) + 1 ensures 1/d < remaining
            # Actually we need 1/d <= remaining, so d >= ceil(1/remaining)
            # But d must be integer.
            # If remaining = 1/5, 1/remaining = 5. d>=5.
            # If remaining = 1/6, 1/remaining = 6. d>=6.
            lower = max(min_d, int(1 / float(remaining)) + 1)
            # Correction: if 1/remaining is integer, say 5, d can be 5.
            # int(1/5) + 1 = 2. This is wrong.
            # We need d such that 1/d <= remaining.
            # d >= 1/remaining.
            # So d >= ceil(1/remaining).
            # If remaining = 1/5, ceil(5) = 5.
            # If remaining = 2/7, 1/remaining = 3.5, ceil = 4.
            
            # Let's calculate precisely
            # d >= 1 / remaining
            # d >= Fraction(1, 1) / remaining
            # d >= 1 / (num/den) = den/num
            num, den = remaining.numerator, remaining.denominator
            lower_exact = (den + num - 1) // num  # ceil(den/num)
            lower = max(min_d, lower_exact)

        # Upper bound from min constraint: remaining - 1/d >= (max_terms-1)/max_d_limit
        thresh = remaining - (max_terms - 1) * Fraction(1, max_d_limit)
        if thresh <= 0:
            return None
        # 1/d <= thresh => d >= 1/thresh
        # Wait, we are looking for d such that remaining - 1/d >= min_remaining
        # 1/d <= remaining - min_remaining
        # d >= 1 / (remaining - min_remaining)
        
        # Also, we must have enough terms left to sum to the remainder.
        # The next denominator d must be small enough so that the remaining sum can be achieved by max_terms-1 terms.
        # Max sum of remaining terms is (max_terms-1)/min_d_next.
        # This is handled by the recursive call's pruning.
        
        # Upper bound on d:
        # We need 1/d <= remaining.
        # Also, the remaining sum (remaining - 1/d) must be achievable by max_terms-1 terms.
        # The maximum possible sum for max_terms-1 terms starting from d+1 is roughly (max_terms-1)/(d+1).
        # So remaining - 1/d <= (max_terms-1)/(d+1) is not a strict upper bound on d,
        # but if d is too large, 1/d is small, remaining is large, and we might not be able to sum it with few terms?
        # No, if d is large, 1/d is small, remaining is close to 1.
        # We need to sum ~1 with max_terms-1 terms.
        # Min denominator is d+1.
        # Max sum is (max_terms-1)/(d+1).
        # So we need remaining - 1/d <= (max_terms-1)/(d+1).
        
        # This inequality helps find an upper bound for d?
        # Actually, if d is very small, 1/d is large, remaining is small.
        # If d is very large, 1/d is small, remaining is large.
        # The constraint is that remaining - 1/d must be <= max_possible_sum(d+1, max_terms-1).
        
        # Let's just use the simple upper bound: d < 1/remaining is not required, d >= 1/remaining is.
        # The loop will terminate when 1/d > remaining? No, we skip if 1/d > remaining.
        # So we can just iterate d from lower to some limit.
        
        # A safe upper limit for d:
        # If we pick d, the remaining sum is R' = remaining - 1/d.
        # We need R' > 0.
        # Also R' must be <= (max_terms-1)/(d+1).
        
        # Let's find max_d such that remaining - 1/d <= (max_terms-1)/(d+1)
        # This is complex. Let's just use a heuristic upper bound.
        # In the previous code, upper was min(max_d_limit, int(1/thresh)).
        # thresh = remaining - (max_terms-1)/max_d_limit.
        # This ensures the *minimum* possible sum of the remaining terms is not exceeded.
        # It doesn't constrain the *maximum* d.
        
        # Actually, if d is too large, 1/d is tiny.
        # The main constraint is that we have enough "space" to fit the remaining sum.
        # But since we are minimizing k, and k is small, d won't be huge.
        
        # Let's stick to the previous upper bound logic which worked for k<8.
        upper_from_min = int(1 / float(thresh))
        upper = min(max_d_limit, upper_from_min)
        
        # Add a check: if upper < lower, return None
        if upper < lower:
            return None

        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return None
            frac = Fraction(1, d)
            if frac > remaining:
                continue
            new_remaining = remaining - frac
            if new_remaining < 0:
                continue
            if max_terms - 1 == 0:
                if new_remaining == 0:
                    return current + [d]
                continue
            result = dfs(new_remaining, d + 1, max_terms - 1, current + [d], max_d_limit)
            if result is not None:
                return result
        return None

    # Search for minimum number of terms
    # We know k < 8 is impossible. Start from 8.
    for k in range(8, 15):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, [], 10000000)
        if result is not None:
            return result

    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
