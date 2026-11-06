import time
import math

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin for 60s limit
    
    N = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 479001600
    # 2: floor(12/2)+floor(12/4)+floor(12/8) = 6+3+1 = 10
    # 3: floor(12/3)+floor(12/9) = 4+1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # 11: floor(12/11) = 1
    
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
    
    # Filter divisors > 12
    # Also, we don't need the divisor N itself as a single factor if we want to minimize max, 
    # but if N is the only option, it's valid. 
    # However, the problem says "1..12 DISTINCT integers". 
    # If we return [N], max is N. 
    # We want to minimize max.
    
    # Filter: d > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    # Precompute exponent vectors for each valid divisor to speed up division check?
    # Or just use modulo on the integer. Modulo on integers is fast.
    
    # We want to find the smallest M in valid_divisors such that 
    # there exists a subset of valid_divisors[:idx_M+1] that multiplies to N.
    
    # To speed up, we can iterate M from the end of a small range? 
    # No, we want MINIMAL M. So we iterate from smallest valid divisor upwards.
    
    # However, if the minimal M is large, we might check many small M's that fail quickly.
    
    def dfs(rem, start_idx, max_idx, count, path):
        if time.time() > deadline:
            return False, None
            
        if rem == 1:
            if count <= 12:
                return True, path[:]
            return False, None
            
        if start_idx > max_idx:
            return False, None
            
        # Pruning: if rem is 1, handled above.
        # If rem is prime, it must be equal to one of the remaining divisors.
        # Check if rem is in valid_divisors[start_idx:max_idx+1]
        # We can maintain a set or just check.
        
        # Pruning: The product of the largest (12-count) remaining divisors must be >= rem
        # This is a weak bound but easy to implement if we have suffix products.
        # Let's skip complex pruning initially and see if it's fast enough.
        
        # Iterate through possible next factors
        # Optimization: We can skip divisors that don't divide rem.
        
        for i in range(start_idx, max_idx + 1):
            d = valid_divisors[i]
            if rem % d == 0:
                # Check if we can form rem/d with divisors from i+1 to max_idx
                # We also need to ensure we don't exceed 12 factors
                if count < 12:
                    path.append(d)
                    found, res = dfs(rem // d, i + 1, max_idx, count + 1, path)
                    if found:
                        return True, res
                    path.pop()
                    
        return False, None

    # Iterate M
    # To be efficient, we can binary search on M? 
    # The property "exists a partition with max <= M" is monotonic.
    # If it works for M, it works for M' > M.
    # So we can binary search for the smallest M.
    
    lo = 0
    hi = len(valid_divisors) - 1
    best_max = None
    best_path = None
    
    # First, check if a solution exists at all (it does, e.g., [13, ...] might not, but [N] works if N>12, which it is).
    # Actually, [N] is a valid solution with 1 factor.
    # So hi is definitely valid.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        M_idx = mid
        # Check if we can solve with max_idx = M_idx
        # We need to find ANY solution, not necessarily the one with minimal number of factors, 
        # just one that exists.
        
        # The DFS above finds the first solution it encounters.
        # We just need to know if one exists.
        
        found, path = dfs(N, 0, M_idx, 0, [])
        
        if found:
            best_max = valid_divisors[M_idx]
            best_path = path
            hi = M_idx - 1
        else:
            lo = M_idx + 1
            
        if time.time() > deadline:
            break

    if best_path is None:
        # Fallback: just return [N] if we timed out or failed logic
        # But [N] should always be found if we search up to N.
        # If binary search failed, maybe try linear from end?
        # Or just return [N] as it is a valid (though likely suboptimal) solution.
        # The task requires minimizing the largest factor.
        # If we time out, we should return the best found.
        # If no best found, return [N].
        return [N]
        
    return best_path
