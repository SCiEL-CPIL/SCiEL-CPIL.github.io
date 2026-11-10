import math
from functools import lru_cache

def construct():
    N = math.factorial(12)
    
    # Get all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Precompute prime factorizations? Not strictly needed for DFS, just modulo.
    
    # We want to find the minimal M such that there exists a subset of 
    # valid_divs <= M that multiplies to N.
    
    # We iterate M from the smallest valid divisor upwards.
    # But M must be at least 13.
    # The candidates for M are the values in valid_divs.
    
    best_solution = None
    best_max = float('inf')
    
    # To speed up, we can just iterate through the possible maximums.
    # The possible maximums are the divisors themselves.
    
    # Let's try each divisor as a potential maximum M.
    # Since we want to minimize M, we check them in ascending order.
    
    # Optimization: 
    # If we find a solution with max M, we can stop.
    # We need to return the list of factors.
    
    # DFS function
    def find_subset(target, max_val, divs):
        # divs is the list of candidates <= max_val
        # We use recursion with index
        n_divs = len(divs)
        
        # Memoization key: (start_index, remaining_target)
        # Since target changes, and index changes.
        # However, target is a divisor of N, so it's from a fixed set of values.
        # We can use a set of visited (index, target) to avoid recomputation.
        visited = set()
        
        def dfs(rem, start_idx, path):
            if rem == 1:
                return True
            if start_idx >= n_divs:
                return False
            if len(path) >= 12:
                return False
            
            # Pruning: 
            # The product of all remaining elements must be >= rem?
            # Actually, we can skip elements. So we just need *some* subset.
            # If the largest element in the remaining list is < rem and rem is prime, fail.
            # Or simpler: if rem is not 1, and no divisor in divs[start_idx:] divides rem, fail.
            # This check is O(n).
            
            # Stronger pruning:
            # Max product possible from remaining:
            # This is expensive to compute every time.
            # Let's stick to simple checks.
            
            key = (start_idx, rem)
            if key in visited:
                return False
            visited.add(key)
            
            for i in range(start_idx, n_divs):
                d = divs[i]
                if d > rem:
                    break # Since divs is sorted, no further d will divide rem
                if rem % d == 0:
                    path.append(d)
                    if dfs(rem // d, i + 1, path):
                        return True
                    path.pop()
            return False
            
        return dfs(target, 0, [])

    # Iterate M
    # We only need to check M values that are in valid_divs.
    # If a solution exists with max M, M must be one of the factors, so M is a divisor.
    
    for M in valid_divs:
        # Candidates are divs <= M
        # Since valid_divs is sorted, we can find the cutoff index
        # Or just pass the whole list and break in DFS if d > M?
        # Better to slice or pass limit.
        
        # Let's pass the list and limit in DFS
        # Modify DFS to take max_val
        
        def find_subset_limited(target, max_val, divs):
            n_divs = len(divs)
            visited = set()
            
            def dfs(rem, start_idx, path):
                if rem == 1:
                    return True
                if start_idx >= n_divs:
                    return False
                if len(path) >= 12:
                    return False
                
                key = (start_idx, rem)
                if key in visited:
                    return False
                visited.add(key)
                
                for i in range(start_idx, n_divs):
                    d = divs[i]
                    if d > max_val:
                        break
                    if d > rem:
                        break
                    if rem % d == 0:
                        path.append(d)
                        if dfs(rem // d, i + 1, path):
                            return True
                        path.pop()
                return False
                
            return dfs(target, 0, [])

        # We need to retrieve the path, not just True/False
        # Let's refactor to return the path
        
        def find_path(target, max_val, divs):
            n_divs = len(divs)
            visited = set()
            path = []
            
            def dfs(rem, start_idx):
                if rem == 1:
                    return True
                if start_idx >= n_divs:
                    return False
                if len(path) >= 12:
                    return False
                
                key = (start_idx, rem)
                if key in visited:
                    return False
                visited.add(key)
                
                for i in range(start_idx, n_divs):
                    d = divs[i]
                    if d > max_val:
                        break
                    if d > rem:
                        break
                    if rem % d == 0:
                        path.append(d)
                        if dfs(rem // d, i + 1):
                            return True
                        path.pop()
                return False
                
            if dfs(target, 0):
                return path
            return None

        sol = find_path(N, M, valid_divs)
        if sol:
            return sol

    # Fallback (should not happen)
    return [N]
