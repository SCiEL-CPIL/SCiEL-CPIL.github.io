from fractions import Fraction
import time

def construct():
    # The minimum number of terms k is 8.
    # We search for a solution with k=8.
    # If not found quickly, we fall back to k=9, etc., but k=8 is guaranteed to exist.
    
    start_time = time.time()
    deadline = 50.0 # 50 seconds margin

    def dfs(rem_num, rem_den, last, depth, k, path):
        # rem = rem_num / rem_den
        if time.time() - start_time > deadline:
            return None
        
        if depth == k:
            if rem_num == 0:
                return path[:]
            return None

        if rem_num <= 0:
            return None
        
        # d must satisfy 1/d <= rem => d >= rem_den / rem_num
        # d >= ceil(rem_den / rem_num)
        min_d = (rem_den + rem_num - 1) // rem_num
        if min_d <= last:
            min_d = last + 1
        
        # d must satisfy: rem - 1/d <= (k - depth - 1) / (d+1)
        # Also rem <= (k - depth) / d => d <= (k - depth) * rem_den / rem_num
        max_d = (k - depth) * rem_den // rem_num
        
        if max_d < min_d:
            return None

        for d in range(min_d, max_d + 1):
            if time.time() - start_time > deadline:
                return None
            
            # new_rem = rem - 1/d = (rem_num * d - rem_den) / (rem_den * d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break
            
            # Simplify fraction to keep numbers small?
            # Or just use the new numerator/denominator.
            # GCD reduction is good for performance.
            from math import gcd
            g = gcd(new_num, rem_den * d)
            new_num //= g
            new_den = (rem_den * d) // g
            
            path.append(d)
            res = dfs(new_num, new_den, d, depth + 1, k, path)
            if res:
                return res
            path.pop()
            
        return None

    # Try k=8
    for k in range(8, 12):
        # Initial remainder is 1/1
        res = dfs(1, 1, 4, 0, k, [])
        if res:
            return res

    # Fallback: return a known valid solution if search fails (should not happen)
    # Or just return the greedy result (might be >8 terms, score < 1.0)
    # But k=8 should be found.
    return []
