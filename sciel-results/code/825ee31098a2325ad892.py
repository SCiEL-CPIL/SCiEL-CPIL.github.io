import time
from fractions import Fraction
from itertools import combinations
import math

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin before 90s kill
    
    # We need to find the minimum number k of distinct denominators in [5, 10000000]
    # such that sum(1/d_i) = 1.
    
    # Lower bound analysis:
    # Max sum with k distinct denominators starting at 5 is sum_{i=0}^{k-1} 1/(5+i).
    # k=8: 1/5+...+1/12 ≈ 0.951 < 1
    # k=9: 1/5+...+1/13 ≈ 0.995 < 1
    # k=10: 1/5+...+1/14 ≈ 1.030 > 1
    # So k >= 10.
    
    # Strategy:
    # 1. Try to find a solution with k=10 using DFS with strong pruning.
    # 2. If that fails or takes too long, try k=11, etc.
    # 3. Fallback to a greedy construction if DFS fails.
    
    def find_solution(k, max_den=10000000):
        """
        Find k distinct denominators in [5, max_den] summing to 1.
        Uses DFS with pruning.
        """
        # Precompute harmonic sums for pruning: H(n) = sum_{i=0}^{n-1} 1/(start+i)
        # We'll compute on the fly but cache the max possible sum for given start and count.
        
        def max_sum_from(start, count):
            """Maximum possible sum of 'count' distinct reciprocals starting from 'start'."""
            s = 0.0
            for i in range(count):
                s += 1.0 / (start + i)
            return s
        
        def dfs(remaining, k_left, min_den, current):
            if time.time() > deadline:
                return None
            
            if k_left == 0:
                if remaining == 0:
                    return current[:]
                return None
            
            if k_left == 1:
                # 1/x = remaining => x = 1/remaining
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        x = den // num
                        if x > min_den and x <= max_den:
                            return current + [x]
                return None
            
            # Bounds for next denominator a
            # 1/a < remaining => a > 1/remaining
            # Also, we need to fit k_left terms.
            # Max sum of k_left terms starting from a is 1/a + 1/(a+1) + ...
            # This must be >= remaining.
            # Min sum of k_left terms starting from a is not directly useful, but:
            # remaining <= k_left / a => a <= k_left / remaining (rough upper bound for a)
            # Actually, a must be such that max_sum_from(a, k_left) >= remaining.
            # And min_sum_from(a, k_left) <= remaining is always true for large a.
            
            # Lower bound for a:
            # 1/a <= remaining => a >= 1/remaining
            # Also a >= min_den
            # So a >= max(min_den, ceil(1/remaining))
            
            # Upper bound for a:
            # max_sum_from(a, k_left) >= remaining
            # Since max_sum_from is decreasing in a, we can find the largest a such that this holds.
            # But we can also use: 1/a + 1/(a+1) + ... + 1/(a+k_left-1) >= remaining
            # A loose upper bound: k_left / a >= remaining => a <= k_left / remaining
            # So a <= floor(k_left / remaining)
            
            lower_a = max(min_den, int(1 / remaining) + 1)
            upper_a = min(max_den, int(k_left / remaining) + 1)
            
            if lower_a > upper_a:
                return None
            
            # Prune: if even taking the smallest possible denominators (lower_a, lower_a+1, ...) the sum is too small
            if max_sum_from(lower_a, k_left) < float(remaining) - 1e-15:
                return None
            
            # Iterate a from lower_a to upper_a
            # But we can prune early: if max_sum_from(a, k_left) < remaining, then for all larger a, it's also < remaining.
            # So we can break.
            
            for a in range(lower_a, upper_a + 1):
                if time.time() > deadline:
                    return None
                
                # Prune: max possible sum from this point
                if max_sum_from(a, k_left) < float(remaining) - 1e-15:
                    break
                
                new_remaining = remaining - Fraction(1, a)
                if new_remaining <= 0:
                    continue
                
                result = dfs(new_remaining, k_left - 1, a + 1, current + [a])
                if result is not None:
                    return result
            
            return None
        
        return dfs(Fraction(1), k, 5, [])

    # Try k=10 first, then 11, etc.
    for k in range(10, 20):
        if time.time() > deadline - 5:
            break
        sol = find_solution(k, max_den=10000000)
        if sol is not None:
            # Verify
            s = sum(Fraction(1, x) for x in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
                return sol

    # Fallback: Greedy algorithm (will produce a valid, but possibly non-minimal, solution)
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0:
            # Find smallest d > last such that 1/d <= remaining
            d = last + 1
            while Fraction(1, d) > remaining:
                d += 1
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
            if len(denoms) > 256:
                break
        return denoms

    sol = greedy_solution()
    s = sum(Fraction(1, x) for x in sol)
    if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
        return sol

    raise RuntimeError("No solution found")
