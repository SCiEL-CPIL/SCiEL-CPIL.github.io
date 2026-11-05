"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin before 90s kill

    def get_divisors(n):
        """Return all divisors of n."""
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)

    def solve_dp(N):
        """Find minimum-size subset of {N//d : d|N, d>=5} that sums to N.
        Returns list of denominators or None."""
        # Weights: for each divisor d of N with d >= 5, weight = N // d
        # We need subset sum = N, minimize count.
        divs = get_divisors(N)
        weights = []
        denom_map = {}
        for d in divs:
            if d >= 5:
                w = N // d
                if w > 0 and w <= N:
                    # Check distinctness: each d gives a unique w? 
                    # w = N//d, so different d can give same w only if N//d1 == N//d2 => d1==d2
                    # So weights are distinct for distinct divisors.
                    if w not in denom_map:
                        denom_map[w] = d
                        weights.append(w)
        
        if not weights:
            return None
        
        # DP: dp[s] = minimum number of items to achieve sum s
        # Initialize with infinity
        INF = float('inf')
        dp = [INF] * (N + 1)
        dp[0] = 0
        # parent tracking: for each state, store (prev_sum, weight_used)
        parent = [None] * (N + 1)
        
        for w in weights:
            # 0/1 knapsack: iterate backwards
            for s in range(N, w - 1, -1):
                if dp[s - w] + 1 < dp[s]:
                    dp[s] = dp[s - w] + 1
                    parent[s] = (s - w, w)
        
        if dp[N] == INF:
            return None
        
        # Reconstruct
        result_weights = []
        s = N
        while s > 0:
            prev_s, w = parent[s]
            result_weights.append(w)
            s = prev_s
        
        # Convert weights back to denominators
        denoms = [denom_map[w] for w in result_weights]
        return denoms

    def verify(denoms):
        """Verify that denoms are distinct, in [5, 10000000], and sum of reciprocals is 1."""
        if not denoms:
            return False
        if len(denoms) != len(set(denoms)):
            return False
        if any(d < 5 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == 1

    # Candidate N values: we want N to have many small divisors (>=5) so that
    # the subset-sum problem is easier and solutions are shorter.
    # Start with highly composite numbers.
    candidates = []
    
    # Generate candidates: multiples that are highly composite
    # Try N = LCM(5,6,7,...,k) for various k, and also other highly composite numbers
    for max_d in range(5, 50):
        # Compute LCM of 5..max_d
        lcm_val = 1
        for i in range(5, max_d + 1):
            lcm_val = lcm_val * i // math.gcd(lcm_val, i)
        if lcm_val <= 10000000 * 100:  # Allow N up to ~1e9 for flexibility
            candidates.append(lcm_val)
    
    # Also add some specific highly composite numbers
    extra = [2520, 5040, 10080, 25200, 50400, 100800, 252000, 504000, 1008000,
             2016000, 5040000, 10080000, 25200000, 50400000, 100800000]
    for e in extra:
        if e not in candidates:
            candidates.append(e)
    
    # Sort by number of divisors >= 5 (more divisors = more options, potentially shorter solutions)
    # But also smaller N = faster DP
    candidates = sorted(set(candidates), key=lambda n: (n, -len([d for d in get_divisors(n) if d >= 5])))
    
    best_solution = None
    best_len = float('inf')
    
    for N in candidates:
        if time.time() > deadline:
            break
        if N > 2000000:  # Cap N to keep DP feasible
            continue
        
        sol = solve_dp(N)
        if sol is not None:
            if verify(sol):
                if len(sol) < best_len:
                    best_len = len(sol)
                    best_solution = sol
                # If we found a solution with 10 terms, that's likely optimal
                if best_len <= 10:
                    break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: DFS search for 10 terms
    def find_k10():
        def dfs(start, remaining, num, den, current):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return current if num == 0 else None
            if num <= 0:
                return None
            
            max_sum = sum(1.0 / (start + i) for i in range(remaining))
            if max_sum < num / den - 1e-12:
                return None
            
            min_d = max(start, (den + num - 1) // num)
            max_d = (den * remaining) // num + 1
            if max_d > 10000000:
                max_d = 10000000
            
            for d in range(min_d, min_d + min(1000, max_d - min_d + 1)):
                if time.time() > deadline:
                    return None
                new_num = num * d - den
                if new_num < 0:
                    break
                if new_num == 0:
                    if remaining == 1:
                        return current + [d]
                    else:
                        continue
                new_den = den * d
                g = math.gcd(new_num, new_den)
                res = dfs(d + 1, remaining - 1, new_num // g, new_den // g, current + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(5, 10, 1, 1, [])
    
    result = find_k10()
    if result and verify(result):
        return result
    
    # Final fallback: greedy construction
    def generate_fallback():
        denoms = []
        target = Fraction(1, 1)
        used = set()
        while target > 0 and len(denoms) < 256:
            min_d = max(5, math.ceil(1.0 / float(target)))
            if min_d < 5:
                min_d = 5
            d = min_d
            while d in used or Fraction(1, d) > target:
                d += 1
            if d > 10000000:
                break
            denoms.append(d)
            used.add(d)
            target -= Fraction(1, d)
        if target == 0 and len(denoms) <= 256:
            return denoms
        return None
    
    result = generate_fallback()
    if result and verify(result):
        return result
    
    return [5, 6, 7, 10, 14, 21, 30, 42, 70, 140]
# EVOLVE-BLOCK-END
