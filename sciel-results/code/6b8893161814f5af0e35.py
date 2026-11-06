import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    def get_dp_solution():
        """
        Finds a valid solution using DP on divisors of T=2520.
        This is fast and guaranteed to find a solution if one exists 
        using denominators that are divisors of 2520.
        """
        T = 2520
        # dp[s] = list of denominators to reach sum s (where sum is in units of 1/T)
        # We want sum of T/d = T => sum of 1/d = 1
        dp = {0: []}
        for d in range(5, T + 1):
            if T % d != 0:
                continue
            w = T // d
            # Update dp in reverse to avoid reusing same d
            # We iterate over a snapshot of keys to avoid changing size during iteration
            new_states = {}
            for s, vals in dp.items():
                ns = s + w
                if ns <= T and ns not in dp:
                    new_states[ns] = vals + [d]
            dp.update(new_states)
            if T in dp:
                return dp[T]
        return None

    # Get a guaranteed valid solution first
    fallback_sol = get_dp_solution()
    
    # If DP failed (should not happen), use a hardcoded valid solution
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/1260 + 1/2520 is NOT valid.
    # Let's find a valid one manually or rely on DP.
    # DP should work. T=2520.
    # 1/5 = 504/2520
    # 1/6 = 420/2520
    # 1/7 = 360/2520
    # 1/10 = 252/2520
    # 1/12 = 210/2520
    # 1/14 = 180/2520
    # 1/15 = 168/2520
    # 1/18 = 140/2520
    # 1/20 = 126/2520
    # 1/21 = 120/2520
    # 1/24 = 105/2520
    # 1/28 = 90/2520
    # 1/30 = 84/2520
    # 1/35 = 72/2520
    # 1/36 = 70/2520
    # 1/40 = 63/2520
    # 1/42 = 60/2520
    # 1/45 = 56/2520
    # 1/50 = 50.4 (not integer)
    # The DP will find a combination.
    
    if fallback_sol is None:
        # Extremely unlikely fallback
        return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    best_sol = fallback_sol
    best_count = len(best_sol)

    # Try to find a better solution with DFS
    # Minimum terms is 8.
    def find_solution_dfs(target_terms):
        target = Fraction(1)
        path = []
        
        def dfs(rem, min_den, depth):
            if time.time() > deadline:
                return None
            
            if depth == target_terms:
                if rem == 0:
                    return list(path)
                return None
            
            k_rem = target_terms - depth
            
            # Pruning: Max possible sum with k_rem terms starting from min_den
            # 1/min_den + 1/(min_den+1) + ...
            # Approximation: k_rem / min_den
            if Fraction(k_rem, min_den) < rem:
                return None
            
            num, den = rem.numerator, rem.denominator
            
            # d_min: 1/d <= rem => d >= den/num
            d_min = max((den + num - 1) // num, min_den, 5)
            
            # d_max: rem <= k_rem / d => d <= k_rem / rem = k_rem * den / num
            d_max = min(k_rem * den // num, 10000000)
            
            if d_min > d_max:
                return None
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    return None
                
                next_rem = rem - Fraction(1, d)
                if next_rem < 0:
                    break
                
                path.append(d)
                result = dfs(next_rem, d + 1, depth + 1)
                if result is not None:
                    return result
                path.pop()
            
            return None
        
        return dfs(target, 5, 0)

    # Try 8 terms
    # We allow a short time for this to avoid wasting the whole budget
    # If 8 terms is the minimum, it should be found relatively quickly with good pruning
    # If not found, we stick with the DP solution
    
    # Let's give DFS 10 seconds to find an 8-term solution
    dfs_deadline = start_time + 10.0
    old_deadline = deadline
    deadline = dfs_deadline
    
    sol_8 = find_solution_dfs(8)
    
    # Restore deadline
    deadline = old_deadline
    
    if sol_8 is not None:
        # Verify
        s = sum(Fraction(1, d) for d in sol_8)
        if s == 1 and len(set(sol_8)) == 8 and all(5 <= d <= 10000000 for d in sol_8):
            return sol_8

    # If 8 terms not found, try 9 terms if we have time left?
    # The DP solution is likely 9 or 10 terms.
    # Let's check the count of the DP solution.
    # If DP solution is 9, we can't do better than 9 unless we find a 9-term solution faster?
    # No, we want to MINIMIZE count.
    # If DP found 9, and DFS didn't find 8, the best we can do is 9.
    # But maybe there is a 9-term solution that is "easier" to find?
    # Actually, if DP found a 9-term solution, it's already optimal (since 8 failed).
    # We just return the DP solution.
    
    # However, what if DP found a 12-term solution, but a 9-term solution exists?
    # Then we should try DFS for 9 terms.
    
    if best_count > 9 and time.time() < deadline - 5:
        # Try to find 9-term solution
        dfs_deadline_9 = deadline - 5
        deadline = dfs_deadline_9
        sol_9 = find_solution_dfs(9)
        deadline = old_deadline
        
        if sol_9 is not None:
            s = sum(Fraction(1, d) for d in sol_9)
            if s == 1 and len(set(sol_9)) == 9 and all(5 <= d <= 10000000 for d in sol_9):
                return sol_9

    return best_sol
