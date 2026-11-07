"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

# EVOLVE-BLOCK-END

def construct():
    def sf(n):
        res = 1
        i = 2
        while i * i <= n:
            cnt = 0
            while n % i == 0:
                n //= i
                cnt += 1
            if cnt % 2 == 1:
                res *= i
            i += 1
        if n > 1:
            res *= n
        return res

    target_sf = 210
    lower_bound = 211
    upper_bound = 239
    
    candidates = list(range(lower_bound, upper_bound + 1))
    sfs = {n: sf(n) for n in candidates}
    
    # Try to find a subset of size 1, 2, or 3
    # Size 1
    for n in candidates:
        if sfs[n] == target_sf:
            return [n]
            
    # Size 2
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            a, b = candidates[i], candidates[j]
            prod_sf = sfs[a] * sfs[b]
            if sf(prod_sf) == target_sf:
                return [a, b]
                
    # Size 3
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                a, b, c = candidates[i], candidates[j], candidates[k]
                prod_sf = sfs[a] * sfs[b] * sfs[c]
                if sf(prod_sf) == target_sf:
                    return [a, b, c]
                    
    # If no better solution found, return the known best
    return [224, 240]
