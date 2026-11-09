import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    MAX_DENOM = 10000000
    MIN_DENOM = 5

    # Known optimal solution for k=7:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/15 = 1
    # Let's verify:
    # LCM of 5,6,7,8,9,10,15 = 2520
    # 2520/5 = 504
    # 2520/6 = 420
    # 2520/7 = 360
    # 2520/8 = 315
    # 2520/9 = 280
    # 2520/10 = 252
    # 2520/15 = 168
    # Sum = 504 + 420 + 360 + 315 + 280 + 252 + 168 = 2300? No.
    # 504+420=924, +360=1284, +315=1599, +280=1879, +252=2131, +168=2299.
    # Wait, 2299 != 2520.
    
    # Let's check the standard identity for 1:
    # 1 = 1/2 + 1/3 + 1/7 + 1/42
    # But denominators must be >= 5.
    
    # Let's try to find a solution with k=5, 6, or 7.
    # If k=5 is impossible, try k=6.
    
    # A known solution for 1 with distinct denominators >= 2:
    # 1 = 1/2 + 1/3 + 1/7 + 1/42 (k=4, but 2<5)
    
    # Let's use the DFS to find the minimal k.
    # The seed program already does this. 
    # The score is 1.0, which means it found a solution. 
    # The task says "minimize their count". 
    # The seed program returns the first solution found for minimal k.
    # If the seed program found k=7, then the score is 1.0 (base).
    # To improve, we need to find a smaller k (if possible) or a faster way to find the minimal k.
    # However, if k=7 is the true minimum, we cannot improve the count.
    # But maybe the seed program is slow and times out on some cases? 
    # Or maybe it finds a non-minimal k because of pruning errors?
    
    # Let's check if k=5 or k=6 is possible.
    # Max sum for k=5 with min denom 5: 1/5+1/6+1/7+1/8+1/9 = 0.2+0.166+0.142+0.125+0.111 = 0.744 < 1.
    # So k=5 is impossible.
    
    # Max sum for k=6 with min denom 5: 1/5+1/6+1/7+1/8+1/9+1/10 = 0.744 + 0.1 = 0.844 < 1.
    # So k=6 is impossible.
    
    # Max sum for k=7 with min denom 5: 0.844 + 1/11 = 0.844 + 0.0909 = 0.935 < 1.
    # So k=7 is impossible??
    
    # Wait, the denominators don't have to be consecutive.
    # But to maximize the sum, we should pick the smallest denominators.
    # So if the sum of the smallest k denominators (5..5+k-1) is less than 1, then k is impossible.
    
    # Let's re-calculate:
    # k=5: 1/5+1/6+1/7+1/8+1/9 = 0.744
    # k=6: +1/10 = 0.844
    # k=7: +1/11 = 0.935
    # k=8: +1/12 = 0.935 + 0.0833 = 1.0183 > 1.
    
    # So k=8 is the minimum possible count!
    # The seed program starts with k=2 and goes up.
    # It will check k=2,3,4,5,6,7 and fail (quickly due to max_sum < 1 check).
    # Then it will try k=8.
    # If it finds a solution for k=8, the answer is 8.
    
    # So the goal is to find a set of 8 distinct integers in [5, 10^7] summing to 1.
    
    def dfs(rem, min_d, depth, path):
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem == 0:
                return path
            return None
        
        if rem == 0:
            return None
        
        # Calculate lower bound for next denominator d
        # 1/d <= rem => d >= 1/rem
        if rem > 0:
            lower_bound = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        else:
            return None
            
        d = lower_bound
        while d <= MAX_DENOM:
            if time.time() > deadline:
                return None
            
            # Check if we can reach 'rem' with 'depth' terms starting from d
            # Max possible sum with 'depth' terms starting from d is sum(1/(d+i) for i in range(depth))
            # If this is < rem, break.
            # For efficiency, we can approximate or just check the first term.
            # Actually, if 1/d + (depth-1)/(d+1) < rem, then no solution.
            # Let's use a quick check.
            if depth == 1:
                if rem == Fraction(1, d):
                    return path + [d]
                else:
                    d += 1
                    continue
            
            # Upper bound check:
            # rem <= 1/d + (depth-1)/(d+1)
            # If rem > 1/d + (depth-1)/(d+1), then d is too small? No, d is too large?
            # Wait, if d increases, 1/d decreases.
            # If d is too large, the max sum decreases.
            # So if max_sum < rem, we should break (d is too large).
            
            # Let's compute max_sum more efficiently.
            # sum(1/(d+i) for i in range(depth))
            # For small depth, this is fine.
            
            max_sum = Fraction(0, 1)
            for i in range(depth):
                max_sum += Fraction(1, d + i)
            
            if max_sum < rem:
                break
                
            # Also, 1/d must be <= rem
            if Fraction(1, d) > rem:
                d += 1
                continue
                
            # Recursive step
            new_rem = rem - Fraction(1, d)
            new_min_d = d + 1
            new_depth = depth - 1
            new_path = path + [d]
            
            res = dfs(new_rem, new_min_d, new_depth, new_path)
            if res is not None:
                return res
            
            d += 1
            
        return None

    # We know k=8 is the minimum.
    # Let's just search for k=8.
    result = dfs(Fraction(1), MIN_DENOM, 8, [])
    if result is not None:
        return result

    # Fallback: Greedy algorithm
    def greedy(rem, min_d):
        path = []
        while rem > 0:
            if time.time() > deadline:
                return None
            d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            if d > MAX_DENOM:
                return None
            path.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        return path

    return greedy(Fraction(1), MIN_DENOM)
