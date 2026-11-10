"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    def solve_for_window(W, m, min_idx):
        values = [None] * n
        values[min_idx] = m
        used = {m}
        
        order = [i for i in range(n) if i != min_idx]
        def domain_size(i):
            d = divisors[i]
            first = m
            if first % d != 0:
                first += d - (first % d)
            if first > m + W:
                return 0
            return (m + W - first) // d + 1
            
        order.sort(key=lambda i: domain_size(i))
        
        def backtrack(k):
            if k == len(order):
                return True
            
            idx = order[k]
            d = divisors[idx]
            
            start_m = m
            if start_m % d != 0:
                start_m += d - (start_m % d)
            
            val = start_m
            while val <= m + W:
                if val not in used:
                    values[idx] = val
                    used.add(val)
                    if backtrack(k + 1):
                        return True
                    used.remove(val)
                    values[idx] = None
                val += d
                
            return False
            
        if backtrack(0):
            return values
        return None

    # Search for smallest W
    for W in range(1, 2000):
        for min_idx in range(n):
            d_min = divisors[min_idx]
            max_m = min(10000, max_val - W)
            if max_m < 1:
                continue
                
            for m in range(1, max_m + 1, d_min):
                feasible = True
                for i in range(n):
                    if i == min_idx: continue
                    d_i = divisors[i]
                    first = m
                    if first % d_i != 0:
                        first += d_i - (first % d_i)
                    if first > m + W:
                        feasible = False
                        break
                if not feasible:
                    continue
                
                res = solve_for_window(W, m, min_idx)
                if res is not None:
                    return [int(x) for x in res]
                    
    # Fallback
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# EVOLVE-BLOCK-END
