"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55

    # Verified known solution with 8 distinct denominators >= 5
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 = 1
    # Let's verify: LCM = 210
    # 42 + 35 + 30 + 21 + 15 + 14 + 10 + 7 = 174... not 210
    # Need to find a correct one.
    
    # Known Egyptian fraction for 1 with denominators >= 5:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/12 + 1/14 + 1/15 + 1/21
    # LCM = 420
    # 84 + 70 + 60 + 42 + 35 + 30 + 28 + 20 = 369 != 420
    
    # Let me compute properly using the DP result
    # For LCM=2520, the DP should find a valid solution
    
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
        for d in candidates:
            w = target_lcm // d
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
                return None
            result.append(d)
            s -= target_lcm // d
            
        return result

    # Try small LCMs first
    for lcm in [2520, 27720, 360360]:
        if time.time() > deadline:
            break
        result = solve_dp(lcm)
        if result is not None:
            # Verify solution
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                total = sum(Fraction(1, d) for d in result)
                if total == 1:
                    return result

    # Fallback: use a verified valid solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 = 174/210, not 1
    # Let me use: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/35
    # LCM=210: 42+35+30+21+15+14+10+6 = 173, not 210
    
    # Use a definitely valid solution:
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/12 + 1/14 + 1/15 + 1/20 + 1/21 + 1/28 + 1/30 + 1/42 = 429/420, not 1
    
    # I'll compute a valid one programmatically if DP fails
    # For now, try DFS as last resort
    def dfs(remaining, min_d, max_terms, current, max_d_limit):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current
        if max_terms == 0:
            return None
        if remaining <= 0:
            return None
        
        max_possible = Fraction(0)
        for i in range(max_terms):
            max_possible += Fraction(1, min_d + i)
        if remaining > max_possible:
            return None
        
        min_possible = max_terms * Fraction(1, max_d_limit)
        if remaining < min_possible:
            return None
        
        num, den = remaining.numerator, remaining.denominator
        lower_exact = (den + num - 1) // num
        lower = max(min_d, lower_exact)
        
        thresh = remaining - (max_terms - 1) * Fraction(1, max_d_limit)
        if thresh <= 0:
            return None
        
        upper_from_min = int(1 / float(thresh))
        upper = min(max_d_limit, upper_from_min)
        
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

    for k in range(8, 15):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, [], 10000000)
        if result is not None:
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                if sum(Fraction(1, d) for d in result) == 1:
                    return result

    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
