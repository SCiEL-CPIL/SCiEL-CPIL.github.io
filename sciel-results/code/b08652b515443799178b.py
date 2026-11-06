import time
from math import gcd

def construct():
    deadline = time.time() + 85
    
    def dfs(num, den, min_d, count, path):
        if time.time() > deadline:
            return None
        if count == 0:
            if num == 0:
                return list(path)
            return None
        if num <= 0:
            return None
            
        # Pruning: Max sum of 'count' terms starting from min_d is count/min_d
        # num/den > count/min_d => num*min_d > den*count
        if num * min_d > den * count:
            return None
            
        # Pruning: Min sum of 'count' terms is count/10000000
        # num/den < count/10000000 => num*10000000 < den*count
        if num * 10000000 < den * count:
            return None
            
        # d_start: 1/d <= num/den => d >= den/num
        d_start = max(min_d, (den + num - 1) // num)
        
        # d_end: 1/d + (count-1)/(d+1) >= num/den
        # Approx d_end = count * den / num
        d_end = int(count * den / num) + 10
        d_end = min(d_end, 10000000)
        
        if d_start > d_end:
            return None

        for d in range(d_start, d_end + 1):
            if time.time() > deadline:
                return None
            
            new_num = num * d - den
            if new_num < 0:
                continue
            new_den = den * d
            
            # Reduce
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            path.append(d)
            result = dfs(new_num, new_den, d + 1, count - 1, path)
            path.pop()
            if result is not None:
                return result
        
        return None

    # Theoretical minimum n is 8.
    # We search for n=8 first.
    result = dfs(1, 1, 5, 8, [])
    if result is not None:
        return result

    # If n=8 fails (should not happen), try larger n
    for n in range(9, 12):
        if time.time() > deadline:
            break
        result = dfs(1, 1, 5, n, [])
        if result is not None:
            return result
            
    raise RuntimeError("no valid representation found")
