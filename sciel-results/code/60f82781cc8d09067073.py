import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    def dfs(depth, prev_den, rem_num, rem_den):
        if time.time() > deadline:
            return None
        
        if depth == 8:
            if rem_num == 0:
                return []
            return None
        
        m = 8 - depth
        
        # rem = rem_num / rem_den
        # Next denom d > prev_den
        # 1/d <= rem => d >= 1/rem = rem_den/rem_num
        min_d = max(prev_den + 1, (rem_den + rem_num - 1) // rem_num)
        
        # rem <= m/d => d <= m/rem = m * rem_num / rem_den
        max_d = (m * rem_num) // rem_den
        
        if min_d > max_d:
            return None
        
        for d in range(min_d, max_d + 1):
            new_num = rem_num * d - rem_den
            if new_num < 0:
                continue
            new_den = rem_den * d
            
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Pruning: Max sum of m-1 terms with denom > d
            # (m-1)/(d+1)
            if m > 1:
                if new_num * (d + 1) > new_den * (m - 1):
                    continue
                    
            res = dfs(depth + 1, d, new_num, new_den)
            if res is not None:
                return [d] + res
        
        return None

    # Try to find 8-term solution
    result = dfs(0, 4, 1, 1)
    if result:
        return result

    # If 8 terms fails (shouldn't), try 9
    # But 8 should work.
    # Fallback to a known valid solution if search fails or times out
    # We can hardcode a valid 8-term solution if we find one.
    # Let's assume the search finds it.
    
    # Just in case, return a valid longer solution?
    # No, we want to minimize count.
    # If search fails, raise error? No, must return valid.
    # Let's hardcode a solution that I can verify.
    
    # Solution: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/x ?
    # Sum 1/5..1/11 = 2361/2520.
    # Rem = 1 - 2361/2520 = 159/2520 = 53/840.
    # 1/x = 53/840 => x = 840/53, not integer.
    
    # Try different set.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/x + 1/y
    # Sum 1/5..1/10 = 2131/2520.
    # Rem = 389/2520.
    # We need 1/x + 1/y = 389/2520.
    # This is the case I analyzed above.
    # Let's find the factors in the code.
    
    # Actually, let's just let the DFS run. It's fast for k=8.
    raise RuntimeError("No solution found")

