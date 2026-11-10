import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin
    
    N = math.factorial(12)
    
    # We search for the minimal M
    # Lower bound for M: 
    # If we use 1 factor, M = N.
    # If we use k factors, M is roughly (N)^(1/k).
    # We iterate M from 13 upwards.
    
    # To speed up, we can bound M.
    # The maximum number of factors is 12.
    # Min possible max factor is at least ceil((N)^(1/12)) but constrained by >12 and distinctness.
    # Let's just iterate M.
    
    def dfs(start, rem, factors):
        if rem == 1:
            return factors[:]
        
        # Pruning: if we can't reach rem with remaining slots
        # Max factors we can add: 12 - len(factors)
        # But we also have the upper bound M.
        
        for i in range(start, M + 1):
            if time.time() > deadline:
                return None
            
            # If i doesn't divide rem, skip
            if rem % i != 0:
                continue
            
            # If i > rem, then rem/i < 1, impossible for integer factors > 1
            # Unless rem == 1, which is handled above.
            if i > rem:
                break
            
            # Pruning: 
            # The smallest product we can form with the remaining (12 - len(factors) - 1) factors
            # is 1 (if we stop). But we must reach rem/i.
            # Actually, we just need to know if rem/i can be formed by distinct integers > i and <= M.
            
            # Stronger pruning:
            # The maximum product we can form with the remaining allowed factors (i+1 to M)
            # is the product of the largest (12 - len(factors) - 1) numbers in that range.
            # If this max product < rem/i, then it's impossible.
            # However, calculating this product every step might be slow if done naively.
            # But M is small.
            
            # Let's check if rem/i is achievable.
            # If rem/i == 1, we found a solution.
            if rem // i == 1:
                return factors + [i]
            
            # If rem/i > 1, we need more factors.
            # The next factor must be > i.
            # If i + 1 > M, we can't pick any more.
            if i == M:
                continue
                
            res = dfs(i + 1, rem // i, factors + [i])
            if res is not None:
                return res
                
        return None

    # Iterate M
    # We can start M from 13.
    # How high can M go? 
    # If we pick just one factor, M = N.
    # But we want to minimize M.
    # A heuristic upper bound: 
    # If we pick 2 factors, M ~ sqrt(N) ~ 21886.
    # If we pick 3 factors, M ~ 782.
    # If we pick 4 factors, M ~ 237.
    # If we pick 5 factors, M ~ 85.
    # If we pick 6 factors, M ~ 37.
    # If we pick 7 factors, M ~ 20.
    # If we pick 8 factors, M ~ 13.
    
    # Since factors must be > 12, the smallest distinct factors are 13, 14, ...
    # Product of 13..19 is ~ 2.5e8. 12! is 4.7e8.
    # So we can likely fit 7 or 8 factors with a small M.
    
    # Let's iterate M from 13 to, say, 500.
    # If not found, expand.
    
    best_result = None
    
    # To be safe, we can also try a greedy approach or other heuristics if DFS is slow, 
    # but for 12! DFS is instant.
    
    # We'll iterate M.
    # To ensure we don't miss a solution with a larger M if a smaller M fails, 
    # we must exhaustively search for each M.
    
    # Optimization: 
    # If for a given M, no solution is found, we increment M.
    
    # What is the max M we should check?
    # If we use 1 factor, M = 479001600.
    # But we want the *minimum* M.
    # We will find it quickly.
    
    # Let's set an upper bound for the loop to prevent infinite loops if logic is flawed,
    # though logically it must terminate.
    # A safe upper bound is N itself.
    
    for M in range(13, 1000): # 1000 is a generous guess, 12!^(1/2) is 21886.
        # Check if time is running out
        if time.time() > deadline:
            break
            
        res = dfs(13, N, [])
        if res is not None:
            best_result = res
            break # Found minimal M
            
    # If not found in 1000, expand search or use fallback
    if best_result is None:
        # Fallback: return a valid solution, e.g., [N]
        # But we want to minimize. 
        # Let's try a few more M values or a different strategy if needed.
        # Given the analysis, M should be very small.
        pass
        
    if best_result is None:
        # Absolute fallback
        return [N]
        
    return best_result
