import math
import time

def construct():
    P = math.factorial(12)
    
    # Get all divisors of P
    divisors = []
    for i in range(1, int(math.isqrt(P)) + 1):
        if P % i == 0:
            divisors.append(i)
            if i != P // i:
                divisors.append(P // i)
    divisors.sort()
    
    best_max = float('inf')
    best_factors = None
    
    # Max k is limited by 13^k <= P
    # 13^7 ~ 6.2e7, 13^8 ~ 8.1e8. 
    # Distinct factors > 12.
    # Min product for k=7: 13*14*15*16*17*18*19 = 87,178,290 < 479,001,600.
    # Min product for k=8: 13*...*20 = 1,743,565,800 > 479,001,600.
    # So k is in [2, 7].
    
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin
    
    def dfs(idx, start_val, remaining_prod, current_factors, k):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
            
        if idx == k:
            x_k = remaining_prod
            # x_k must be >= start_val (which is > x_{k-1})
            if x_k >= start_val:
                # x_k is automatically a divisor of P
                # Check distinctness: ensured by start_val > x_{k-1}
                if x_k < best_max:
                    best_max = x_k
                    best_factors = current_factors + [x_k]
            return
        
        # We need to pick x_idx such that:
        # 1. x_idx divides remaining_prod
        # 2. x_idx >= start_val
        # 3. x_idx < remaining_prod (since there are other factors)
        # 4. x_idx^(k - idx + 1) < remaining_prod (pruning)
        
        # Pre-compute the max allowed value for x_idx based on power
        # x_idx < remaining_prod^(1/(k - idx + 1))
        power = k - idx + 1
        max_val = int(remaining_prod ** (1.0 / power))
        
        # Iterate through divisors
        # We can optimize by only looking at divisors of remaining_prod
        # But divisors list is small (792), so just iterate and filter
        for d in divisors:
            if d < start_val:
                continue
            if d > max_val:
                break
            if d >= remaining_prod:
                break
            if remaining_prod % d != 0:
                continue
            
            # Additional pruning:
            # The remaining product is rem = remaining_prod // d
            # We need k - idx factors, all > d
            # So rem must be > d * (d+1) * ... * (d + (k - idx - 1))
            # This is a strong bound.
            min_rem_prod = 1
            for j in range(k - idx):
                min_rem_prod *= (d + 1 + j)
                if min_rem_prod > remaining_prod // d:
                    break
            
            if min_rem_prod > remaining_prod // d:
                # Even with the smallest possible distinct factors, product is too large
                # Since d is increasing, this will hold for all larger d too?
                # Yes, if d increases, min_rem_prod increases.
                # And remaining_prod // d decreases.
                # So we can break.
                break
            
            dfs(idx + 1, d + 1, remaining_prod // d, current_factors + [d], k)

    for k in range(2, 8):
        dfs(1, 13, P, [], k)
        
    if best_factors is None:
        # Fallback, though shouldn't happen
        return [P]
        
    return best_factors
