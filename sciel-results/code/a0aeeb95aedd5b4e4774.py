def construct():
    from fractions import Fraction
    import math
    import time
    
    start_time = time.time()
    timeout = 55  # seconds, leave 5s margin
    
    target = Fraction(5, 31)
    
    # Current best: [8, 28, 1736]
    best_max = 1736
    best_sol = [8, 28, 1736]
    
    def dfs(rem, min_d, cur, max_d):
        nonlocal best_max, best_sol
        if time.time() - start_time > timeout:
            return
        if not rem:
            if max_d < best_max:
                best_max = max_d
                best_sol = cur[:]
            return
        if len(cur) >= 256:
            return
        if max_d >= best_max:
            return
        
        # Pruning: max possible remaining sum
        max_terms_left = 256 - len(cur)
        if max_terms_left / min_d < rem:
            return
        
        d_start = max(min_d, math.ceil(1/rem))
        
        for d in range(d_start, min(d_start + 1000, best_max)):
            if time.time() - start_time > timeout:
                return
            frac = Fraction(1, d)
            if frac > rem:
                break
            new_rem = rem - frac
            new_max = max(max_d, d)
            if new_max >= best_max:
                continue
            cur.append(d)
            dfs(new_rem, d + 1, cur, new_max)
            cur.pop()
    
    dfs(target, 2, [], 0)
    
    return best_sol
