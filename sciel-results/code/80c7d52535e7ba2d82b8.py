from fractions import Fraction
import time

def construct():
    target = Fraction(1, 1)
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # We determined k must be at least 8.
    # Let's try to find a solution with k=8.
    # If not found quickly, try k=9, etc.
    
    def dfs(rem, prev_d, depth, path, k_target):
        if time.time() > deadline:
            return None
        
        if depth == k_target:
            if rem == 0:
                return path[:]
            return None
        
        if rem <= 0:
            return None
            
        m = k_target - depth + 1
        # Lower bound for d:
        # 1/d <= rem => d >= 1/rem
        # Also d > prev_d
        d_low = max(prev_d + 1, -(-rem.denominator // rem.numerator))
        
        # Upper bound for d:
        # The sum of the next m terms is at most m/d (if all were 1/d, but they must be distinct and larger, so sum < m/d).
        # Actually, sum_{i=0}^{m-1} 1/(d+i) <= m/d.
        # So we need m/d >= rem => d <= m/rem.
        # To be safe, d_high = floor(m * rem.denominator / rem.numerator)
        d_high = (m * rem.denominator) // rem.numerator
        
        if d_low > d_high:
            return None

        for d in range(d_low, d_high + 1):
            # Pruning:
            # 1. Check if max possible sum of remaining m-1 terms is >= new_rem
            # Max sum of m-1 distinct terms starting from d+1 is sum(1/(d+1) ... 1/(d+m-1))
            # This is an upper bound. If this upper bound < new_rem, we can't reach rem.
            # Since we iterate d upwards, new_rem decreases, but the max possible sum of future terms also decreases.
            # Actually, if we pick a smaller d, we get a larger 1/d, leaving a smaller new_rem.
            # The condition "max future sum >= new_rem" is the critical one.
            # If it fails for a given d, it will fail for all larger d (because new_rem gets smaller? No, 1/d gets smaller, so new_rem = rem - 1/d gets LARGER. And max future sum gets SMALLER. So if it fails for d, it fails for d+1).
            
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
                
            if m == 1:
                if new_rem == 0:
                    return path + [d]
                continue
            
            # Check if we can fill the rest
            # Max sum of m-1 terms starting from d+1
            # We can approximate or calculate exactly. For small m (<= 8), exact is fast.
            max_future = sum(Fraction(1, d + j) for j in range(1, m))
            if max_future < new_rem:
                break # No need to check larger d
            
            # Also check min sum?
            # Min sum of m-1 terms is approaching 0, so no lower bound check needed for possibility,
            # except that new_rem must be positive (checked) and representable.
            
            res = dfs(new_rem, d, depth + 1, path + [d], k_target)
            if res:
                return res
        
        return None

    # Try k=8
    result = dfs(target, 4, 1, [], 8)
    if result:
        return result
    
    # If k=8 fails (shouldn't, but just in case), try k=9
    if time.time() < deadline - 5:
        result = dfs(target, 4, 1, [], 9)
        if result:
            return result
            
    # Fallback: return a known valid solution if found in literature or precomputed
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/... 
    # Let's just return an empty list or raise error if not found, but k=8 should work.
    # A known 8-term solution:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/...
    # Actually, let's just let the DFS find it. It's very fast.
    
    return []
