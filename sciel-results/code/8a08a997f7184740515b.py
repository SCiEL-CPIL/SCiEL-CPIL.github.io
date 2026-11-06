import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    target = Fraction(5, 31)
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target

    def check_3term(limit):
        """
        Returns a 3-term solution [a, b, c] with max denom <= limit, or None.
        """
        # 1/a + 1/b + 1/c = 5/31
        # a < b < c
        # a must be >= 7
        # c <= limit
        # b <= limit
        # a < limit
        
        for a in range(7, min(1000, limit)):  # a is usually small
            if time.time() - start_time > time_limit:
                return None
            
            rem = target - Fraction(1, a)
            if rem <= 0:
                continue
            
            # 1/b + 1/c = rem = p/q
            # p = rem.numerator, q = rem.denominator
            p = rem.numerator
            q = rem.denominator
            
            # Reduce p/q is already done by Fraction
            # (p*b - q)*(p*c - q) = q^2
            # Let d1 = p*b - q, d2 = p*c - q
            # d1*d2 = q^2
            # b = (d1+q)/p, c = (d2+q)/p
            # b, c must be integers, distinct, > a, <= limit
            
            q_sq = q * q
            # d1 must be a divisor of q_sq
            # d1 < d2 => d1 < q
            # Also b > a => (d1+q)/p > a => d1 > p*a - q
            # c <= limit => (d2+q)/p <= limit => d2 <= p*limit - q
            
            lower_d1 = p * a - q + 1
            if lower_d1 < 1:
                lower_d1 = 1
            
            upper_d1 = q # since d1 < d2 and d1*d2=q^2, d1 < q
            
            # Iterate d1
            # To speed up, we can iterate divisors of q^2
            # But q is small (denom of 5/31 - 1/a).
            # For a=7, rem=2/217. q=217. q^2 ~ 47000.
            # Iterating up to 217 is trivial.
            
            for d1 in range(lower_d1, upper_d1 + 1):
                if q_sq % d1 != 0:
                    continue
                
                d2 = q_sq // d1
                
                # Check integrality
                if (d1 + q) % p != 0:
                    continue
                if (d2 + q) % p != 0:
                    continue
                
                b = (d1 + q) // p
                c = (d2 + q) // p
                
                if b <= a or c <= b:
                    continue
                
                if b <= limit and c <= limit:
                    return [a, b, c]
        
        return None

    def check_4term(limit):
        """
        Returns a 4-term solution with max denom <= limit, or None.
        Uses DFS with pruning.
        """
        def dfs(rem, min_d, terms_left, current):
            if time.time() - start_time > time_limit:
                return None
            
            if terms_left == 0:
                if rem == 0:
                    return current
                return None
            
            if rem == 0:
                return None # Must use distinct denominators, can't add 0
            
            # Lower bound for d: 1/d <= rem => d >= 1/rem
            # Also d >= min_d
            lower_bound = (rem.denominator + rem.numerator - 1) // rem.numerator
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Upper bound for d:
            # We need to be able to form rem with terms_left terms, all <= limit.
            # Max possible sum with terms_left terms starting from lower_bound is 
            # 1/lb + 1/(lb+1) + ...
            # But a tighter bound: 
            # The smallest possible sum with terms_left terms is terms_left / limit.
            # So rem must be >= terms_left / limit.
            if rem < Fraction(terms_left, limit):
                return None
            
            # Also, the largest term 1/d must be <= rem.
            # And the sum of the remaining terms must be >= rem - 1/d.
            # For the last term (terms_left == 1):
            if terms_left == 1:
                # rem must be 1/d for some integer d <= limit
                if rem.denominator % rem.numerator == 0:
                    d = rem.denominator // rem.numerator
                    if d >= min_d and d <= limit:
                        return current + [d]
                return None
            
            # Iterate d
            # Heuristic: Try smaller d first? Or larger?
            # Smaller d reduces rem faster.
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                # 1/d <= rem
                if Fraction(1, d) > rem:
                    continue
                
                next_rem = rem - Fraction(1, d)
                if next_rem < 0:
                    continue
                
                res = dfs(next_rem, d + 1, terms_left - 1, current + [d])
                if res is not None:
                    return res
            
            return None
        
        return dfs(target, 1, 4, [])

    # Binary search for minimal K
    # We know K=1736 works.
    # Check if smaller K works.
    
    # First, quickly check if a 3-term solution exists for K < 1736
    # We can just run check_3term with decreasing K?
    # Or binary search.
    
    low = 1
    high = best_max_denom
    
    # Optimization: check 3-term solutions first in a loop for decreasing K
    # But binary search is safer.
    
    # Actually, let's just try to find a better 3-term solution first.
    # The current check_3term(1736) will find [8, 28, 1736].
    # We want to know if check_3term(K) works for K < 1736.
    
    # Let's do a linear descent from 1736 downwards, checking 3-term first, then 4-term.
    # Since 3-term is very fast, we can afford to check many K values.
    # But 4-term is slow. We should only call 4-term if 3-term fails for a K.
    
    current_limit = best_max_denom - 1
    
    while current_limit >= 7:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        sol_3 = check_3term(current_limit)
        if sol_3 is not None:
            best_solution = sol_3
            best_max_denom = max(sol_3)
            current_limit = best_max_denom - 1
            continue
        
        # 3-term failed. Try 4-term.
        # To avoid excessive 4-term searches, only try every N steps or if we have time.
        # Given the scores, the optimal is likely close to the current best.
        # Let's try 4-term.
        sol_4 = check_4term(current_limit)
        if sol_4 is not None:
            best_solution = sol_4
            best_max_denom = max(sol_4)
            current_limit = best_max_denom - 1
            continue
        
        # If both fail, no solution exists for this limit?
        # Not necessarily, 5-term might exist. But 4-term is a good bet.
        # If we assume 3 or 4 terms are optimal, we can break.
        # But to be safe, we might just break and return best found.
        # Or we can continue decreasing limit, hoping 3-term works later?
        # No, if 3-term fails for K, it might work for K+1?
        # Yes, because a solution with max denom K+1 might have a term K+1.
        # So if check_3term(K) fails, check_3term(K-1) might also fail.
        # Actually, if a solution exists with max denom <= K, it also exists with max denom <= K+1.
        # So feasibility is monotonic.
        # If check_3term(K) fails AND check_4term(K) fails, it doesn't mean no solution exists with max <= K.
        # It just means no 3 or 4 term solution.
        # But we can't search 5 terms easily.
        # So we break and assume current best is optimal for 3/4 terms.
        break

    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
