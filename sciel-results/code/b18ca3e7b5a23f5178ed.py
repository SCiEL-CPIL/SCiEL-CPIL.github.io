import math
import bisect

def construct():
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    def find_factors(M):
        # Returns list of factors or None
        # Uses DFS with pruning
        def dfs(last_factor, current_prod, count, path):
            if current_prod == N:
                if count > 0:
                    return path[:]
                else:
                    return None
            
            if count >= 12:
                return None
            
            remaining = N // current_prod
            if remaining == 1:
                return None
            
            # Pruning: 
            # The remaining factors must be > last_factor and <= M.
            # There are (12 - count) slots left.
            # Min possible value for the next factor is last_factor + 1.
            # Min possible product for the remaining (12-count) factors:
            # (last_factor + 1) * (last_factor + 2) * ... * (last_factor + (12-count))
            # If remaining < this min product, impossible.
            
            min_prod = 1
            for j in range(1, 13 - count):
                min_prod *= (last_factor + j)
            if remaining < min_prod:
                return None
            
            # Max possible product for the remaining (12-count) factors:
            # Each is <= M.
            # If remaining > M^(12-count), impossible.
            if remaining > (M ** (12 - count)):
                return None
            
            # Iterate possible next factors
            start_idx = bisect.bisect_right(valid_divisors, last_factor)
            end_idx = bisect.bisect_right(valid_divisors, M)
            
            for i in range(start_idx, end_idx):
                d = valid_divisors[i]
                if d > remaining:
                    break
                if remaining % d == 0:
                    new_remaining = remaining // d
                    new_count = count + 1
                    new_last = d
                    
                    # Additional pruning before recursion
                    # Check min product for the rest
                    min_prod_rest = 1
                    for j in range(1, 13 - new_count):
                        min_prod_rest *= (new_last + j)
                    if new_remaining < min_prod_rest:
                        continue
                    if new_remaining > (M ** (12 - new_count)):
                        continue
                    
                    path.append(d)
                    res = dfs(new_last, current_prod * d, new_count, path)
                    if res is not None:
                        return res
                    path.pop()
            return None
            
        return dfs(12, 1, 0, [])

    # Iterate M from smallest to largest
    for M in valid_divisors:
        sol = find_factors(M)
        if sol:
            return sol
            
    # Fallback
    return [N]
