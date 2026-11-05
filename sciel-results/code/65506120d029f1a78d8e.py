from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = 50 # seconds
    
    def dfs(rem_num, rem_den, last, depth, k, path):
        if time.time() - start_time > deadline:
            return None
        
        # Normalize rem
        g = gcd(rem_num, rem_den)
        rem_num //= g
        rem_den //= g
        
        if depth == k:
            if rem_num == 0:
                return path[:]
            return None
        
        # Pruning: If rem is 0, we have no sum left, but need to place more terms.
        if rem_num == 0:
            return None
        
        r = k - depth
        
        # Pruning: Max possible sum of remaining terms
        # The largest term is 1/(last+1).
        # So sum <= r / (last+1).
        # If rem > r / (last+1), impossible.
        if rem_num * (last + 1) > rem_den * r:
            return None
        
        # Bounds for next denominator d
        # 1/d <= rem => d >= 1/rem = rem_den / rem_num
        min_d = rem_den // rem_num
        if min_d <= last:
            min_d = last + 1
            
        # 1/d >= rem / r => d <= r / rem = r * rem_den / rem_num
        max_d = (r * rem_den) // rem_num
        
        if min_d > max_d:
            return None
        
        for d in range(min_d, max_d + 1):
            # Check if 1/d <= rem
            if rem_num * d < rem_den:
                continue
                
            # New rem = rem - 1/d
            # rem = rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num < 0:
                continue
                
            path.append(d)
            res = dfs(new_num, new_den, d, depth + 1, k, path)
            if res:
                return res
            path.pop()
        
        return None

    # Try k from 5 upwards to minimize count
    for k in range(5, 12):
        res = dfs(1, 1, 4, 0, k, [])
        if res:
            return res
            
    # Fallback to greedy with integer math
    target_num, target_den = 1, 1
    denominators = []
    last = 4
    while target_num > 0:
        d = target_den // target_num
        if d <= last:
            d = last + 1
        while target_num * d < target_den:
            d += 1
        denominators.append(d)
        # target = target - 1/d
        target_num = target_num * d - target_den
        target_den = target_den * d
        last = d
        if len(denominators) > 256:
            break
    return denominators
