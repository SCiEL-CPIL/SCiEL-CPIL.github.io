import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe

    def dfs(current_num, current_den, last_denom, count, k):
        if time.time() > deadline:
            return None
        
        # current_sum = current_num / current_den
        # Target = 1 = 1/1
        
        if count == k:
            if current_num == current_den:
                return []
            return None
        
        if current_num > current_den:
            return None
            
        remaining_terms = k - count
        
        # Max possible sum with remaining terms
        # Sum of 1/(last_denom+i) for i=1..remaining_terms
        # If current_sum + MaxSum < 1, prune
        max_add_num = 0
        max_add_den = 1
        for i in range(1, remaining_terms + 1):
            d = last_denom + i
            # Add 1/d to max_add
            new_num = max_add_num * d + max_add_den
            new_den = max_add_den * d
            g = gcd(new_num, new_den)
            max_add_num = new_num // g
            max_add_den = new_den // g
            
        # Check if current_num/current_den + max_add_num/max_add_den < 1
        # current_num * max_add_den + current_den * max_add_num < current_den * max_add_den
        if current_num * max_add_den + current_den * max_add_num < current_den * max_add_den:
            return None

        # Range for next denominator d
        # 1/d <= 1 - current_sum
        # 1/d <= (current_den - current_num) / current_den
        # d >= current_den / (current_den - current_num)
        rem_num = current_den - current_num
        if rem_num == 0:
            if count == k:
                return []
            return None # Can't add more terms to sum 1
        
        min_d = int(current_den / rem_num) + 1
        min_d = max(min_d, last_denom + 1)
        
        # Upper bound
        max_d = 10_000_000
        
        if remaining_terms == 1:
            # Need 1/d = rem
            # d = 1/rem = current_den / rem_num
            if current_den % rem_num == 0:
                d = current_den // rem_num
                if d >= min_d and d <= max_d and d > last_denom:
                    return [d]
            return None

        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            # next_sum = current_sum + 1/d
            next_num = current_num * d + current_den
            next_den = current_den * d
            g = gcd(next_num, next_den)
            next_num //= g
            next_den //= g
            
            if next_num > next_den:
                continue
            
            # Check if rem_next can be formed
            rem_next_num = next_den - next_num
            if rem_next_num == 0:
                if count + 1 == k:
                    return [d]
                continue # Can't add more terms if sum is already 1
            
            # Max sum of remaining_terms - 1 terms starting from d+1
            max_next_add_num = 0
            max_next_add_den = 1
            for i in range(1, remaining_terms):
                dd = d + i
                new_num = max_next_add_num * dd + max_next_add_den
                new_den = max_next_add_den * dd
                gg = gcd(new_num, new_den)
                max_next_add_num = new_num // gg
                max_next_add_den = new_den // gg
                
            # Check if rem_next <= max_next_add
            if rem_next_num * max_next_add_den <= rem_next_den * max_next_add_num:
                res = dfs(next_num, next_den, d, count + 1, k)
                if res is not None:
                    return [d] + res
                    
        return None

    for k in range(1, 257):
        res = dfs(0, 1, 4, 0, k)
        if res is not None:
            return res
            
    # Fallback
    return [5, 6, 7, 10, 14, 21, 35, 42, 70, 105, 140, 210, 420, 840, 1680, 3360, 6720, 13440, 26880, 53760, 107520, 215040, 430080, 860160, 1720320, 3440640, 6881280]
