import time
from fractions import Fraction
from math import gcd, log, ceil

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We stop searching at 80s.
    time_limit = 80.0
    
    target = Fraction(5, 31)
    
    # Initial known valid solution
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
        Fast check for 3-term solution with max denom <= limit.
        Returns [a, b, c] or None.
        """
        # 1/a + 1/b + 1/c = 5/31
        # a < b < c
        # a must be >= 7 (since 1/7 < 5/31 < 1/6)
        # Also 3/a > 5/31 => a < 93/5 = 18.6, so a is in [7, 18]
        # Wait, a doesn't have to be that small if b and c are large?
        # No, 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7.
        # Also 1/a + 1/b + 1/c > 3/c. If c <= limit, 3/limit < 5/31 => limit > 93/5 = 18.6.
        # So limit must be at least 19 for any solution.
        
        for a in range(7, min(limit, 100)): # a is usually small
            if time.time() - start_time > time_limit:
                return None
            
            rem = target - Fraction(1, a)
            if rem <= 0:
                continue
            
            p = rem.numerator
            q = rem.denominator
            
            # 1/b + 1/c = p/q
            # (p*b - q)(p*c - q) = q^2
            # d1 = p*b - q, d2 = p*c - q
            # d1 * d2 = q^2
            # b = (d1 + q) / p, c = (d2 + q) / p
            # b > a => d1 > p*a - q
            # c <= limit => d2 <= p*limit - q
            # Also d1 < d2 => d1 < q
            
            lower_d1 = p * a - q + 1
            if lower_d1 < 1:
                lower_d1 = 1
            
            upper_d1 = q # since d1 < d2 and product is q^2, d1 < q
            
            # Optimization: Iterate divisors of q^2? Or just range?
            # q is small. For a=7, rem=2/217, q=217. q^2 ~ 47000.
            # Iterating range(1, 218) is very fast.
            
            for d1 in range(lower_d1, upper_d1 + 1):
                if q_sq_check(d1, q, p, a, limit):
                    # Found d1
                    d2 = (q * q) // d1
                    if (d1 + q) % p != 0 or (d2 + q) % p != 0:
                        continue
                    b = (d1 + q) // p
                    c = (d2 + q) // p
                    if b <= a or c <= b:
                        continue
                    if b <= limit and c <= limit:
                        return [a, b, c]
        return None

    def q_sq_check(d1, q, p, a, limit):
        # Helper to avoid recalculating q*q in inner loop if needed, 
        # though q is fixed per 'a'. 
        # Actually, just do the math in the loop for simplicity, it's fast enough.
        return (q * q) % d1 == 0

    def harmonic_approx(n):
        if n <= 0:
            return 0
        # H_n approx ln(n) + gamma + 1/(2n)
        gamma = 0.5772156649015329
        return log(n) + gamma + 1.0/(2.0*n)

    def check_dfs(rem, min_d, terms_left, current, limit):
        """
        DFS to find a solution with max denom <= limit.
        Returns list of denominators or None.
        """
        if time.time() - start_time > time_limit:
            return None
            
        if terms_left == 0:
            if rem == 0:
                return current
            return None
            
        if rem == 0:
            return None # Must use distinct denominators, can't add 0 terms to make 0 if we need more terms? 
                        # Actually if rem==0 and terms_left>0, we failed because we must use distinct denoms 
                        # and we can't add 1/d = 0. So this branch is dead.
        
        if rem < 0:
            return None
            
        # Pruning: Max possible sum of remaining terms
        # We need to pick 'terms_left' distinct denominators from [min_d+1, limit]
        # The maximum sum is obtained by picking the smallest denominators: 
        # 1/(min_d+1) + 1/(min_d+2) + ... 
        # Approximation: integral of 1/x from min_d to limit
        if terms_left > 0:
            # If even the sum of the 'terms_left' smallest available reciprocals is < rem, impossible
            # Smallest available are min_d+1, min_d+2, ...
            # Sum approx: log(limit) - log(min_d) ... roughly
            # More precise bound: 
            # If limit < min_d + terms_left - 1, we don't have enough numbers.
            if limit < min_d + terms_left - 1:
                return None
            
            # Upper bound on sum: sum(1/x for x in range(min_d+1, min_d+terms_left+1))
            # This is the MAX possible sum for the remaining terms.
            # If rem > max_sum, impossible.
            max_sum = 0
            # To be fast, use harmonic approx for the sum of the first 'terms_left' terms starting from min_d+1
            # Sum_{i=1}^{k} 1/(min_d+i)
            # If k is small, just loop. If k is large, approx.
            # Since max terms is likely small (< 10), loop is fine.
            for i in range(1, terms_left + 1):
                d_val = min_d + i
                if d_val > limit:
                    break
                max_sum += Fraction(1, d_val)
            
            if rem > max_sum:
                return None

        # Lower bound for next denominator d:
        # 1/d <= rem => d >= 1/rem
        lower_bound = (rem.denominator + rem.numerator - 1) // rem.numerator
        if lower_bound < min_d + 1:
            lower_bound = min_d + 1
            
        if lower_bound > limit:
            return None
            
        # Upper bound for d:
        # We must be able to form rem - 1/d with (terms_left - 1) terms from [d+1, limit].
        # The max sum of remaining (terms_left-1) terms is approx log(limit) - log(d).
        # If rem - 1/d > max_remaining_sum, then d is too small (1/d is too big? No, 1/d small means d big).
        # Wait. If d is small, 1/d is large. rem - 1/d is small.
        # If d is large, 1/d is small. rem - 1/d is large.
        # The constraint is that rem - 1/d must be achievable by the remaining terms.
        # The MIN possible sum of remaining terms is (terms_left-1)/limit.
        # So rem - 1/d >= (terms_left-1)/limit => 1/d <= rem - (terms_left-1)/limit
        # d >= 1 / (rem - (terms_left-1)/limit)
        
        # Let's stick to iterating d from lower_bound to limit.
        # Heuristic: Try larger d first? 
        # If we want to minimize the LARGEST denominator, we are already bounded by 'limit'.
        # Finding ANY solution within 'limit' is the goal for the feasibility check.
        # Usually, smaller d leads to larger remaining rem, which is harder to satisfy with large denominators?
        # Actually, for feasibility, we just need one path.
        
        for d in range(lower_bound, limit + 1):
            if time.time() - start_time > time_limit:
                return None
                
            next_rem = rem - Fraction(1, d)
            
            if next_rem < 0:
                # 1/d > rem. Since d is increasing, 1/d is decreasing.
                # So next_rem will increase. We can't break, we must continue.
                # But if d is very large, next_rem becomes positive.
                # However, if 1/d > rem, we can't use this d.
                continue
            
            if next_rem == 0:
                if terms_left == 1:
                    return current + [d]
                else:
                    # We have 0 remainder but still terms to fill. Impossible with distinct positive denoms.
                    continue
            
            # Recurse
            res = check_dfs(next_rem, d, terms_left - 1, current + [d], limit)
            if res is not None:
                return res
                
        return None

    def is_feasible(K):
        """
        Checks if a solution exists with max denominator <= K.
        Returns the solution if found, else None.
        """
        # First, try fast 3-term check
        sol_3 = check_3term(K)
        if sol_3 is not None:
            return sol_3
            
        # If 3-term failed, try 4-term (and potentially more, but 4 is likely enough for small K)
        # We restrict DFS depth to avoid exponential blowup.
        # If K is small, the search space is small.
        # If K is large, the search space is huge, but we have time limits.
        
        # Try 4 terms
        sol_4 = check_dfs(target, 1, 4, [], K)
        if sol_4 is not None:
            return sol_4
            
        # Try 5 terms? Only if we have plenty of time left.
        if time.time() - start_time < time_limit - 10:
            sol_5 = check_dfs(target, 1, 5, [], K)
            if sol_5 is not None:
                return sol_5
                
        return None

    # Binary Search for minimal K
    # We know K=1736 is feasible.
    # We want to find the smallest K.
    # Feasibility is monotonic: if feasible(K), then feasible(K+1).
    
    low = 1
    high = best_max_denom
    
    # We can optimize the search space.
    # 3-term solution [8, 28, 1736]. 
    # Is there a 3-term solution with smaller max?
    # check_3term(K) is fast. We can binary search specifically for 3-term solutions first.
    
    # Let's binary search for the minimal K where ANY solution (3, 4, or 5 term) exists.
    
    # Note: The "best" solution is the one with the minimum max denominator.
    # We are looking for the smallest K such that is_feasible(K) is True.
    
    # To save time, we can first check if a 3-term solution exists for K < 1736.
    # If yes, we binary search for the min 3-term K.
    # If no, we binary search for the min K where a 4-term solution exists.
    
    # Strategy:
    # 1. Binary search for minimal K_3 (3-term solution).
    # 2. Binary search for minimal K_4 (4-term solution).
    # 3. The answer is min(K_3, K_4).
    # However, K_4 search is expensive.
    
    # Better Strategy:
    # Binary search on K in [1, 1736].
    # In the middle, run is_feasible(K).
    # If feasible, high = K.
    # If not, low = K + 1.
    
    # Optimization: 
    # If K < 19, no solution is possible (3/19 < 5/31).
    low = 19
    
    while low < high:
        if time.time() - start_time > time_limit - 5:
            break
            
        mid = (low + high) // 2
        
        sol = is_feasible(mid)
        
        if sol is not None:
            # Update best solution if this one is better (smaller max denom)
            # Actually, mid is the bound. The actual max denom of sol might be <= mid.
            # We want to minimize the actual max denom.
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                
            # Since a solution exists for mid, it exists for any K >= mid.
            # We want the smallest K, so we try to go lower.
            high = mid
        else:
            low = mid + 1

    # After binary search, 'low' is the smallest K we found feasible (or close to it).
    # But we should verify the final candidate.
    # It's possible the binary search stopped early due to time.
    # We should return the best_solution found.
    
    # One final check: did we find a solution better than the initial one?
    # The initial one is [8, 28, 1736].
    
    if verify(best_solution):
        return best_solution
    
    # Fallback to known valid solution
    return [8, 28, 1736]

# Run construct to test? No, the evaluator calls construct().
# But we can add a small self-test if __name__ == '__main__'
if __name__ == '__main__':
    sol = construct()
    print(f"Solution: {sol}")
    print(f"Max Denom: {max(sol)}")
    s = sum(Fraction(1, d) for d in sol)
    print(f"Sum: {s} (Target: 5/31)")
    print(f"Valid: {s == Fraction(5, 31)}")
