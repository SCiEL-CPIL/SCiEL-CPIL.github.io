# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from fractions import Fraction
import sys
import random

# Try to import scipy MIP for potential use
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def construct():
    start_time = time.time()
    deadline = start_time + 58
    
    # 1. Try to find k=8 solution using optimized DFS
    # The theoretical minimum for denominators >= 5 is 8.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = ?
    # Sum = 0.2 + 0.1666 + 0.1428 + 0.125 + 0.1111 + 0.1 + 0.0909 + 0.0833 ≈ 1.0198 > 1.
    # So 8 terms is possible. We need to search for it.
    
    # Strategy: 
    # - Use DFS with strong pruning.
    # - Pruning: 
    #   1. Sum so far + max possible sum of remaining terms >= 1
    #   2. Sum so far + min possible sum of remaining terms <= 1
    #   3. Denominators must be distinct and increasing.
    #   4. If remaining terms = 1, check if 1/denom is integer.
    #   5. If remaining terms = 2, solve for two fractions exactly.
    
    result = _search_k(8, deadline)
    if result:
        return result
        
    # If k=8 failed, try k=9
    if time.time() < deadline - 2:
        result = _search_k(9, deadline)
        if result:
            return result
            
    # Fallback: Hardcoded valid solution or DP
    return _fallback()

def _search_k(k, deadline):
    """
    Search for k distinct denominators d1 < d2 < ... < dk such that sum(1/di) = 1.
    di >= 5.
    """
    d = []
    
    # Helper for pruning: max sum of m distinct denominators starting from last_denom+1
    def max_possible_sum(remaining_terms, last_denom):
        # Sum of 1/(last_denom + i) for i=1..remaining_terms
        # Approximation: remaining_terms / (last_denom + 1)
        # But this is an overestimate if denominators can be larger.
        # Actually, we want to know if it's possible to reach 1.
        # If current_sum + max_possible < 1, prune.
        # max_possible is achieved by smallest denominators.
        # sum_{i=1}^{m} 1/(last_denom + i)
        # We can compute this exactly for small m.
        s = 0.0
        for i in range(1, remaining_terms + 1):
            s += 1.0 / (last_denom + i)
        return s
    
    # Helper for pruning: min sum of m distinct denominators starting from last_denom+1
    def min_possible_sum(remaining_terms, last_denom):
        # This is not useful for lower bound, but we need to ensure we don't exceed 1.
        # Actually, we iterate denominators, so we naturally stop if sum > 1.
        pass
    
    def dfs(idx, num, den, last_denom):
        if time.time() > deadline:
            return None
            
        if idx == k:
            if num == den:
                return d[:]
            return None
            
        remaining_terms = k - idx
        remaining_num = den - num
        remaining_den = den
        
        if remaining_num <= 0:
            return None
            
        # Pruning 1: Can we reach 1?
        # We need sum_{i=0}^{remaining_terms-1} 1/d_{i} = remaining_num / remaining_den
        # The maximum sum we can add is with the smallest available denominators.
        # d_{idx} > last_denom
        # Max sum = sum_{i=1}^{remaining_terms} 1/(last_denom + i)
        max_sum = max_possible_sum(remaining_terms, last_denom)
        # remaining_num / remaining_den must be <= max_sum
        # remaining_num * (last_denom + 1) <= remaining_den * remaining_terms (approx)
        # Let's use exact fraction comparison if possible, or float for speed.
        if remaining_num * (last_denom + 1) > remaining_den * remaining_terms:
            # This is a loose bound. 1/(last_denom+1) * remaining_terms is an overestimate of the sum of reciprocals of larger denominators.
            # Wait. 1/(last_denom+1) > 1/(last_denom+2) > ...
            # So sum_{i=1}^m 1/(last_denom+i) < m / (last_denom+1).
            # If remaining_num/remaining_den > m / (last_denom+1), it's impossible.
            # So if remaining_num * (last_denom + 1) > remaining_den * remaining_terms, return None.
            return None
        
        # Determine range for next denominator d_val
        # 1/d_val <= remaining_num / remaining_den
        # d_val >= remaining_den / remaining_num
        min_d_float = remaining_den / remaining_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Upper bound for d_val:
        # We need to be able to complete the sum with remaining_terms - 1 terms.
        # The minimum sum of the remaining_terms - 1 terms is with the largest denominators? No.
        # We want to know if 1/d_val is too small.
        # If 1/d_val + max_sum(remaining_terms - 1, d_val) < remaining_num / remaining_den, then d_val is too small? No.
        # If 1/d_val + max_sum(remaining_terms - 1, d_val) < target, then even with smallest next denominators we can't reach target.
        # So d_val is too small? No, if d_val is small, 1/d_val is large.
        # If 1/d_val is too large, we might exceed 1? No, we check new_num > new_den.
        
        # Let's derive a better upper bound for d_val.
        # We need 1/d_val + sum_{j=1}^{remaining_terms-1} 1/d_{j} = remaining_num / remaining_den
        # Since d_j > d_val, sum_{j} 1/d_j < (remaining_terms - 1) / (d_val + 1)
        # So 1/d_val + (remaining_terms - 1) / (d_val + 1) >= remaining_num / remaining_den
        # Let f(x) = 1/x + (remaining_terms - 1) / (x + 1)
        # f(x) is decreasing for x > 0.
        # We need f(d_val) >= remaining_num / remaining_den
        # So d_val <= f^{-1}(remaining_num / remaining_den)
        
        # Solve for x: 1/x + (m-1)/(x+1) = T
        # (x+1 + x(m-1)) / (x(x+1)) = T
        # (mx + 1) / (x^2 + x) = T
        # T x^2 + (T - m) x - 1 = 0
        # x = [-(T-m) + sqrt((T-m)^2 + 4T)] / (2T)
        
        m = remaining_terms
        T = remaining_num / remaining_den
        # T = remaining_num / remaining_den
        # Coefficients: A=T, B=T-m, C=-1
        # Discriminant D = (T-m)^2 + 4T
        A = T
        B = T - m
        C = -1
        D = B*B - 4*A*C
        if D < 0:
            max_d = min_d + 100 # Fallback
        else:
            x = (-B + math.sqrt(D)) / (2*A)
            max_d = int(math.floor(x))
        
        # Add a safety margin
        max_d = min(max_d + 10, 10000000)
        
        if min_d > max_d:
            return None
            
        # Special case: 1 term left
        if remaining_terms == 1:
            if remaining_den % remaining_num == 0:
                d_val = remaining_den // remaining_num
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    return d[:]
            return None
            
        # Special case: 2 terms left
        if remaining_terms == 2:
            p = remaining_num
            q = remaining_den
            # 1/a + 1/b = p/q
            # q(a+b) = pab
            # (pa-q)(pb-q) = q^2
            # X = pa - q, Y = pb - q. XY = q^2.
            # a = (X+q)/p, b = (Y+q)/p.
            # X must be divisor of q^2.
            # X < q (since a < b => X < Y => X^2 < q^2 => X < q)
            # Also a > last_denom => (X+q)/p > last_denom => X > p*last_denom - q
            
            # Iterate divisors X of q^2 such that X < q
            # To speed up, iterate i from 1 to sqrt(q^2) = q
            # But q can be large.
            # Optimization: Only check divisors X where (X+q) % p == 0.
            # This is hard to filter.
            
            # Alternative: Iterate a from last_denom+1 to ...
            # 1/a < p/q => a > q/p
            # 1/a > p/q - 1/a_max? No.
            # b = qa / (pa - q)
            # b > a => qa > a(pa-q) => q > pa - q => 2q > pa => a < 2q/p
            # So a is in (q/p, 2q/p).
            min_a = max(last_denom + 1, int(math.ceil(q / p)))
            max_a = int(math.floor(2 * q / p))
            
            for a in range(min_a, max_a + 1):
                if time.time() > deadline:
                    return None
                num_b = q * a
                den_b = p * a - q
                if den_b <= 0:
                    continue
                if num_b % den_b != 0:
                    continue
                b = num_b // den_b
                if b <= a:
                    continue
                if b > 10000000:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) == Fraction(p, q):
                    d.append(a)
                    d.append(b)
                    return d[:]
            return None
        
        # Iterate d_val
        # Optimize iteration order? Maybe larger denominators first?
        # Smaller denominators lead to larger sums, so they are more likely to be valid.
        # But we want to find A solution.
        # Let's iterate from min_d to max_d.
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
                
            if d_val <= last_denom:
                continue
                
            # new_sum = num/den + 1/d_val
            new_num = num * d_val + den
            new_den = den * d_val
            
            # Check if new_sum > 1
            if new_num > new_den:
                # Since d_val is increasing, new_sum is decreasing.
                # So we can break?
                # new_sum = num/den + 1/d_val. As d_val increases, 1/d_val decreases.
                # So if new_num > new_den, it means sum > 1.
                # For larger d_val, sum will be smaller.
                # So we should NOT break, we should continue to larger d_val.
                continue
                
            d.append(d_val)
            res = dfs(idx + 1, new_num, new_den, d_val)
            if res:
                return res
            d.pop()
            
        return None
        
    return dfs(0, 0, 1, 4)

def _fallback():
    """
    Fallback to a known valid solution or DP.
    """
    # DP over divisors of LCM(5..N) is too slow for large N.
    # But we can use a pre-computed solution.
    # Known solution for k=9:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 ?
    # Let's check a known identity.
    # 1 = 1/2 + 1/3 + 1/6 (denoms < 5, invalid)
    # 1 = 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/14 + 1/42 ?
    # Denoms: 3,4,5,6,7,14,42. Min denom 3.
    # We need min denom >= 5.
    
    # Let's use the DP fallback from the seed, which finds a solution for target 2520.
    # It might take a bit of time but is robust.
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    # If DP fails, return a hardcoded large solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 ...
    # Let's just return a valid set found via quick search if DP is slow.
    # But DP on 2520 is fast.
    
    raise RuntimeError("No valid solution found")

# EVOLVE-BLOCK-END
