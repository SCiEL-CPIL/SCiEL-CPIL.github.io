import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Precompute all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
    
    # Filter divisors > 12
    candidates = [d for d in divisors if d > 12]
    candidates.sort()
    
    # We want to minimize the largest factor M.
    # Lower bound for M:
    # 12!^(1/12) is approx 47.
    # But factors must be distinct and > 12.
    # Let's start from a reasonable lower bound.
    # If we use 12 factors, the geometric mean is ~47.
    # If we use fewer, the mean is higher.
    # Let's try M from 47 upwards.
    
    best_factors = [N]
    best_max = N
    
    # To avoid re-running DFS for very small M that are impossible,
    # we can just iterate.
    
    # Memoization for DFS might help if we hit same (target, start_idx) states,
    # but start_idx depends on the specific M's candidate list.
    # Since M increases, the candidate list grows.
    
    # We will iterate M.
    # To be safe with time, we check time periodically.
    
    # Optimization: 
    # The DFS is exponential in the worst case, but the number of factors is small (<=12).
    # The branching factor is the number of divisors of 'target' that are in the candidate list.
    
    def dfs(target, start_idx, current_path, max_allowed):
        if time.time() - start_time > 50:
            return None
        
        if target == 1:
            if len(current_path) <= 12:
                return list(current_path)
            return None
            
        if len(current_path) >= 12:
            return None
            
        # Pruning: 
        # The remaining target must be 1 after using at most (12 - len(current_path)) factors.
        # The maximum possible product of these factors is max_allowed^(12 - len(current_path)).
        # If target > max_allowed^(12 - len(current_path)), it's impossible.
        # Also, target must be divisible by at least one candidate >= start_idx.
        
        remaining_slots = 12 - len(current_path)
        if target > max_allowed ** remaining_slots:
            return None
            
        # Check if target can be formed by any subset of candidates[start_idx:]
        # Simple check: target must be divisible by some candidate.
        # We iterate.
        
        for i in range(start_idx, len(candidates)):
            c = candidates[i]
            if c > max_allowed:
                break
            if c > target:
                break
            if target % c != 0:
                continue
                
            current_path.append(c)
            res = dfs(target // c, i + 1, current_path, max_allowed)
            if res is not None:
                return res
            current_path.pop()
            
        return None

    # Iterate M
    # Start from a lower bound.
    # 12!^(1/12) = 47.
    # But we need distinct factors > 12.
    # Let's start from 47.
    
    # However, there is a chance that a solution with M < 47 exists if we use fewer factors?
    # No, geometric mean argument: if we use k factors, max >= (12!)^(1/k).
    # For k=1, max=12!.
    # For k=2, max >= 21886.
    # For k=3, max >= 782.
    # ...
    # For k=12, max >= 47.
    # So the global minimum must be at least 47.
    
    # We should check k=12, then k=11, etc?
    # No, minimizing max factor.
    # A solution with k=12 might have max 50.
    # A solution with k=11 might have max 51.
    # A solution with k=10 might have max 56.
    # So we just iterate M from 47 upwards.
    # The first M for which DFS returns a solution is the optimal M?
    # Not necessarily. DFS finds *a* solution for that M.
    # Since we iterate M in increasing order, the first M that yields a valid solution
    # corresponds to the minimum possible maximum factor.
    # Why? If there is a solution with max factor M*, then for any M >= M*,
    # that same solution is valid (since all factors <= M* <= M).
    # So the set of valid M is an interval [M_min, infinity).
    # Thus, finding the smallest M with a valid solution gives the optimal answer.
    
    for M in range(47, N + 1):
        if time.time() - start_time > 50:
            break
            
        # Optimization: If M is small, the candidate list is small.
        # We can just run DFS.
        
        res = dfs(N, 0, [], M)
        if res is not None:
            return res
            
    return best_factors
