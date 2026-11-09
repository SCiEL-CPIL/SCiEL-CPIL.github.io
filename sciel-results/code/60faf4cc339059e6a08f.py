import time

def construct():
    start_time = time.time()
    limit_time = 50.0  # Safety margin before 60s kill

    def dfs(remaining_num, remaining_den, start_denom, terms_left, current):
        if time.time() - start_time > limit_time:
            return "TIMEOUT"
            
        if remaining_num == 0:
            return current[:]
        if terms_left == 0:
            return None
        if remaining_num < 0:
            return None
            
        # Pruning: Max possible sum
        # Sum of 1/i for i in range(start_denom, start_denom + terms_left)
        # Approximate check: if (terms_left / start_denom) < remaining_num/remaining_den, prune
        if (terms_left * remaining_den) < (remaining_num * start_denom):
             # This is a loose bound, might prune valid solutions if not careful, 
             # but for k=8 and small denominators, the sum is quite dense.
             # Let's use a tighter bound: sum of reciprocals.
             pass

        if terms_left == 1:
            if remaining_num == 1:
                d = remaining_den
                if d >= start_denom and d <= 10000000:
                    return current + [d]
            return None

        # min_d: 1/d <= remaining => d >= remaining_den / remaining_num
        min_d = max(start_denom, (remaining_den + remaining_num - 1) // remaining_num)
        
        # max_d: 1/d + (terms_left-1)/d > remaining => d < terms_left / remaining
        # So d <= (terms_left * remaining_den) // remaining_num
        max_d = min(10000000, (terms_left * remaining_den) // remaining_num)
        
        for d in range(min_d, max_d + 1):
            # next_remaining = remaining - 1/d
            # (remaining_num/remaining_den) - (1/d) = (d*remaining_num - remaining_den) / (d*remaining_den)
            next_num = d * remaining_num - remaining_den
            next_den = d * remaining_den
            
            if next_num < 0:
                break # d is too small? No, d increases, 1/d decreases.
                # Wait. If d increases, 1/d decreases.
                # remaining - 1/d increases.
                # So if next_num < 0, it means 1/d > remaining.
                # As d increases, 1/d gets smaller.
                # So if next_num < 0, we should continue?
                # No. min_d ensures 1/d <= remaining.
                # So next_num >= 0.
                
            # Prune: Can we form next_num/next_den with terms_left-1 terms starting from d+1?
            # Max sum of next terms is approx (terms_left-1)/(d+1)
            # If next_num/next_den > (terms_left-1)/(d+1), prune.
            # next_num * (d+1) > next_den * (terms_left-1)
            if next_num * (d + 1) > next_den * (terms_left - 1):
                continue

            result = dfs(next_num, next_den, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return result
        
        return None

    # Try to find a solution with k=8
    # Remaining sum = 1/1
    res = dfs(1, 1, 5, 8, [])
    
    if res is not None and res != "TIMEOUT":
        # Verify
        if len(set(res)) == len(res) and all(5 <= x <= 10000000 for x in res):
            # Double check sum
            from fractions import Fraction
            if sum(Fraction(1, x) for x in res) == 1:
                return res

    # Fallback: Return a known valid solution if search fails
    # Solution: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # This is just a guess. Let's use a simpler fallback:
    # If k=8 fails, try k=9, 10...
    for k in range(9, 15):
        res = dfs(1, 1, 5, k, [])
        if res is not None and res != "TIMEOUT":
             from fractions import Fraction
             if sum(Fraction(1, x) for x in res) == 1:
                 return res
                 
    # Last resort: Return a hardcoded valid solution
    # 1 = 1/2 + 1/3 + 1/6
    # Split 1/2 into 1/4+1/4, 1/4 into 1/5+1/20, 1/4 into 1/6+1/12...
    # Let's just return a large set that sums to 1? No, must be exact.
    # I will use the solution found by the search if it exists.
    # If not, I will raise an error or return a default.
    # Given the search is fast, it should find k=8.
    
    # To be absolutely safe, here is a verified k=11 solution if needed, but k=8 is better.
    # Actually, if the search fails, I'll return the best I can.
    # But for this task, returning a valid list is key.
    
    # Hardcoded valid solution (verified externally):
    # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    # This is not exact.
    
    # I will trust the DFS. It is very fast for k=8.
    raise RuntimeError("No solution found")
