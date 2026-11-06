import random
import time

def construct():
    start_time = time.time()
    timeout = 55.0
    max_size = 48
    max_val = 512
    
    def is_valid(S):
        n = len(S)
        if n < 2:
            return None
        S_sorted = sorted(S)
        total_sum = sum(S_sorted)
        
        for a in S_sorted:
            others_sum = total_sum - a
            if others_sum < 2 * a:
                continue
                
            others = [y for y in S_sorted if y != a]
            m = len(others)
            
            dp = [0] * (m + 1)
            dp[0] = 1
            mask = (1 << (others_sum + 1)) - 1
            
            for y in others:
                for k in range(m, 0, -1):
                    dp[k] = (dp[k] | (dp[k-1] << y)) & mask
            
            for k in range(2, m + 1):
                target = k * a
                if target > others_sum:
                    break
                if (dp[k] >> target) & 1:
                    return a
        return None

    for restart in range(100):
        S = set(random.sample(range(1, max_val + 1), max_size))
        
        for _ in range(500):
            if time.time() - start_time > timeout:
                return list(S)
                
            violator = is_valid(list(S))
            if violator is None:
                return list(S)
                
            new_val = random.randint(1, max_val)
            attempts = 0
            while new_val in S and attempts < 50:
                new_val = random.randint(1, max_val)
                attempts += 1
            if new_val in S:
                continue
                
            S.remove(violator)
            S.add(new_val)
            
    return list(S)
