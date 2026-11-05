import math
import time

def construct():
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Map for quick lookup if a number is a divisor
    # Not strictly needed if we filter the sorted list, but good for O(1) check
    # Actually, filtering the sorted list of 900 items is fast enough.
    
    def dfs(rem, last, M, path):
        # rem: remaining product to form
        # last: the previous factor chosen, next must be > last
        # M: maximum allowed factor
        # path: list of chosen factors so far
        
        if rem == 1:
            return True
        
        # Check if rem itself is a valid next factor (and the last one)
        # rem must be a divisor of rem (trivially true), 
        # and last < rem <= M
        if last < rem <= M:
            # We also need to ensure rem divides N? 
            # Since rem is derived from N by dividing by previous factors, 
            # and we only divide by divisors, rem is always a divisor of N.
            # So yes, rem is a valid factor.
            path.append(rem)
            return True
            
        # Find candidates for the next factor
        # Candidates must be divisors of rem
        # And last < candidate <= M
        # To keep factors distinct and sorted, we pick the next factor > last.
        
        # Optimization: 
        # If rem is prime, it must be the last factor. 
        # We already checked if rem <= M. If rem > M, and rem is prime, 
        # we can't form it with factors <= M (unless we split it, but primes can't be split).
        # So if rem is prime and rem > M, return False.
        # If rem is prime and rem <= M, we returned True above.
        
        # Iterate through divisors of rem
        # We can precompute divisors of N, but rem changes.
        # It's faster to find divisors of rem on the fly or filter the global divisors.
        # Since divisors of rem are a subset of divisors of N, 
        # we can iterate through the global divisors list and check if rem % d == 0.
        
        # To prune better, we should only consider d such that last < d <= M.
        # Also, d must be <= rem.
        
        # We can break early if d > rem or d > M.
        
        for d in divisors:
            if d <= last:
                continue
            if d > M:
                break
            if d > rem:
                break
            if rem % d == 0:
                path.append(d)
                if dfs(rem // d, d, M, path):
                    return True
                path.pop()
                
        return False

    # Binary search for the minimum M
    low = 13  # Factors must be > 12
    high = N   # Trivial solution [N]
    
    best_path = [N] # Fallback
    
    # Time guard
    start_time = time.time()
    
    while low < high:
        if time.time() - start_time > 50:
            break
            
        mid = (low + high) // 2
        
        path = []
        if dfs(N, 12, mid, path):
            high = mid
            best_path = path[:] # Save the best path found so far with smaller M
        else:
            low = mid + 1
            
    # If the loop finished or broke, ensure we have a valid path for 'low'
    # The binary search converges to the smallest M where dfs is True.
    # If we broke due to time, 'low' might not be the true minimum, but best_path is valid for some M <= current high.
    # However, the binary search logic `high = mid` ensures `high` always points to a feasible M.
    # So `best_path` corresponds to the last feasible `mid` found.
    # To be safe, if we broke early, we might want to verify the path for the final `high` or just return best_path.
    # Actually, if we break, `best_path` is a valid solution.
    # If we finish, `low == high`, and we should run dfs one last time to get the path for `low` (which is the optimal M).
    
    if time.time() - start_time < 55:
        final_path = []
        if dfs(N, 12, low, final_path):
            best_path = final_path
        else:
            # This shouldn't happen if logic is correct, but fallback to previous best
            pass
            
    return best_path
